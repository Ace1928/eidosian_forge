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
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from . import __version__
from .configuration_utils import PretrainedConfig
from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .debug_utils import DebugOption, DebugUnderflowOverflow
from .feature_extraction_sequence_utils import SequenceFeatureExtractor
from .hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from .integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from .integrations.tpu import tpu_spmd_dataloader
from .modelcard import TrainingSummary
from .modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from .models.auto.modeling_auto import (
from .optimization import Adafactor, get_scheduler
from .pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_callback import (
from .trainer_pt_utils import (
from .trainer_utils import (
from .training_args import OptimizerNames, ParallelMode, TrainingArguments
from .utils import (
from .utils.quantization_config import QuantizationMethod
def compare_trainer_and_checkpoint_args(self, training_args, trainer_state):
    attributes_map = {'logging_steps': 'logging_steps', 'eval_steps': 'eval_steps', 'save_steps': 'save_steps'}
    has_warning = False
    warning_str = 'Warning: The following arguments do not match the ones in the `trainer_state.json` within the checkpoint directory: '
    for arg_attr, state_attr in attributes_map.items():
        arg_value = getattr(training_args, arg_attr, None)
        state_value = getattr(trainer_state, state_attr, None)
        if arg_value is not None and state_value is not None and (arg_value != state_value):
            warning_str += f'\n\t{arg_attr}: {arg_value} (from args) != {state_value} (from trainer_state.json)'
            has_warning = True
    train_bs_args = training_args.per_device_train_batch_size
    train_bs_state = trainer_state.train_batch_size // max(1, training_args.n_gpu)
    if train_bs_args != train_bs_state:
        warning_str += f'\n\tper_device_train_batch_size: {train_bs_args} (from args) != {train_bs_state} (from trainer_state.json)'
        has_warning = True
    if has_warning:
        logger.warning_once(warning_str)