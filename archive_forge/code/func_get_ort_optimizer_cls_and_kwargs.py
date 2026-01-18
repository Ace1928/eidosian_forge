import functools
import math
import os
import shutil
import sys
import time
import types
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.integrations import hp_params
from transformers.utils import is_accelerate_available
from packaging import version
import huggingface_hub.utils as hf_hub_utils
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, RandomSampler
from transformers.data.data_collator import DataCollator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_zero3_enabled
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback, TrainerState
from transformers.trainer_pt_utils import (
from transformers.trainer_utils import (
from transformers.training_args import ParallelMode
from transformers.utils import (
from ..utils import logging
from .training_args import ORTOptimizerNames, ORTTrainingArguments
from .utils import (
@staticmethod
def get_ort_optimizer_cls_and_kwargs(args: ORTTrainingArguments) -> Tuple[Any, Any]:
    """
        Returns the optimizer class and optimizer parameters implemented in ONNX Runtime based on `ORTTrainingArguments`.

        Args:
            args (`ORTTrainingArguments`):
                The training arguments for the training session.
        """
    optimizer_kwargs = {'lr': args.learning_rate}
    adam_kwargs = {'betas': (args.adam_beta1, args.adam_beta2), 'eps': args.adam_epsilon}
    if args.optim == ORTOptimizerNames.ADAMW_ORT_FUSED:
        try:
            from onnxruntime.training.optim import FusedAdam
            optimizer_cls = FusedAdam
            optimizer_kwargs.update(adam_kwargs)
        except ImportError:
            raise ImportError('ORTTrainer tried to instantiate ORT FusedAdam but onnxruntime-training is not correctly installed!')
    else:
        raise ValueError(f'ORTTrainer cannot instantiate unsupported optimizer: {args.optim}')
    return (optimizer_cls, optimizer_kwargs)