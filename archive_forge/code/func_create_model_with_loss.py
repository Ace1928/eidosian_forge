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
def create_model_with_loss(self):
    model_with_loss = ModuleWithLoss(self.model, self.args, self.label_smoother)
    model_with_loss.compute_model_plus_loss_internal = types.MethodType(Trainer.compute_loss, model_with_loss)
    return model_with_loss