import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from .integrations import (
import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from . import __version__
from .configuration_utils import PretrainedConfig
from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .debug_utils import DebugOption, DebugUnderflowOverflow
from .hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from .integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from .integrations.tpu import tpu_spmd_dataloader
from .modelcard import TrainingSummary
from .modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from .models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from .optimization import Adafactor, get_scheduler
from .pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_callback import (
from .trainer_pt_utils import (
from .trainer_utils import (
from .training_args import OptimizerNames, ParallelMode, TrainingArguments
from .utils import (
from .utils.quantization_config import QuantizationMethod
@staticmethod
def get_optimizer_cls_and_kwargs(args: TrainingArguments) -> Tuple[Any, Any]:
    """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`transformers.training_args.TrainingArguments`):
                The training arguments for the training session.

        """
    optim_args = {}
    if args.optim_args:
        for mapping in args.optim_args.replace(' ', '').split(','):
            key, value = mapping.split('=')
            optim_args[key] = value
    optimizer_kwargs = {'lr': args.learning_rate}
    adam_kwargs = {'betas': (args.adam_beta1, args.adam_beta2), 'eps': args.adam_epsilon}
    if args.optim == OptimizerNames.ADAFACTOR:
        optimizer_cls = Adafactor
        optimizer_kwargs.update({'scale_parameter': False, 'relative_step': False})
    elif args.optim == OptimizerNames.ADAMW_HF:
        from .optimization import AdamW
        optimizer_cls = AdamW
        optimizer_kwargs.update(adam_kwargs)
    elif args.optim in [OptimizerNames.ADAMW_TORCH, OptimizerNames.ADAMW_TORCH_FUSED]:
        from torch.optim import AdamW
        optimizer_cls = AdamW
        optimizer_kwargs.update(adam_kwargs)
        if args.optim == OptimizerNames.ADAMW_TORCH_FUSED:
            optimizer_kwargs.update({'fused': True})
    elif args.optim == OptimizerNames.ADAMW_TORCH_XLA:
        try:
            from torch_xla.amp.syncfree import AdamW
            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        except ImportError:
            raise ValueError('Trainer failed to import syncfree AdamW from torch_xla.')
    elif args.optim == OptimizerNames.ADAMW_TORCH_NPU_FUSED:
        try:
            from torch_npu.optim import NpuFusedAdamW
            optimizer_cls = NpuFusedAdamW
            optimizer_kwargs.update(adam_kwargs)
        except ImportError:
            raise ValueError('Trainer failed to import FusedAdamW from torch_npu.')
    elif args.optim == OptimizerNames.ADAMW_APEX_FUSED:
        try:
            from apex.optimizers import FusedAdam
            optimizer_cls = FusedAdam
            optimizer_kwargs.update(adam_kwargs)
        except ImportError:
            raise ValueError('Trainer tried to instantiate apex FusedAdam but apex is not installed!')
    elif args.optim in [OptimizerNames.ADAMW_BNB, OptimizerNames.ADAMW_8BIT, OptimizerNames.PAGED_ADAMW, OptimizerNames.PAGED_ADAMW_8BIT, OptimizerNames.LION, OptimizerNames.LION_8BIT, OptimizerNames.PAGED_LION, OptimizerNames.PAGED_LION_8BIT, OptimizerNames.RMSPROP_BNB, OptimizerNames.RMSPROP_8BIT, OptimizerNames.RMSPROP_32BIT]:
        try:
            from bitsandbytes.optim import AdamW, Lion, RMSprop
            is_paged = False
            optim_bits = 32
            optimizer_cls = None
            additional_optim_kwargs = adam_kwargs
            if 'paged' in args.optim:
                is_paged = True
            if '8bit' in args.optim:
                optim_bits = 8
            if 'adam' in args.optim:
                optimizer_cls = AdamW
            elif 'lion' in args.optim:
                optimizer_cls = Lion
                additional_optim_kwargs = {'betas': (args.adam_beta1, args.adam_beta2)}
            elif 'rmsprop' in args.optim:
                optimizer_cls = RMSprop
                additional_optim_kwargs = optim_args
            bnb_kwargs = {'optim_bits': optim_bits}
            if 'rmsprop' not in args.optim:
                bnb_kwargs['is_paged'] = is_paged
            optimizer_kwargs.update(additional_optim_kwargs)
            optimizer_kwargs.update(bnb_kwargs)
        except ImportError:
            raise ValueError('Trainer tried to instantiate bnb optimizer but bnb is not installed!')
        if is_bitsandbytes_available() and version.parse(importlib.metadata.version('bitsandbytes')) < version.parse('0.41.1'):
            logger.warning('You are using 8-bit optimizers with a version of `bitsandbytes` < 0.41.1. It is recommended to update your version as a major bug has been fixed in 8-bit optimizers.')
    elif args.optim == OptimizerNames.ADAMW_ANYPRECISION:
        try:
            from torchdistx.optimizers import AnyPrecisionAdamW
            optimizer_cls = AnyPrecisionAdamW
            optimizer_kwargs.update(adam_kwargs)
            optimizer_kwargs.update({'use_kahan_summation': strtobool(optim_args.get('use_kahan_summation', 'False')), 'momentum_dtype': getattr(torch, optim_args.get('momentum_dtype', 'float32')), 'variance_dtype': getattr(torch, optim_args.get('variance_dtype', 'float32')), 'compensation_buffer_dtype': getattr(torch, optim_args.get('compensation_buffer_dtype', 'bfloat16'))})
        except ImportError:
            raise ValueError('Please install https://github.com/pytorch/torchdistx')
    elif args.optim == OptimizerNames.SGD:
        optimizer_cls = torch.optim.SGD
    elif args.optim == OptimizerNames.ADAGRAD:
        optimizer_cls = torch.optim.Adagrad
    elif args.optim == OptimizerNames.RMSPROP:
        optimizer_cls = torch.optim.RMSprop
    else:
        raise ValueError(f'Trainer cannot instantiate unsupported optimizer: {args.optim}')
    return (optimizer_cls, optimizer_kwargs)