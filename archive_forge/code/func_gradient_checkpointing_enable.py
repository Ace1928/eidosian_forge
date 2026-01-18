import collections
import copy
import functools
import gc
import importlib.metadata
import inspect
import itertools
import json
import os
import re
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from zipfile import is_zipfile
import torch
from packaging import version
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, Identity
from torch.utils.checkpoint import checkpoint
from .activations import get_activation
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import GenerationConfig, GenerationMixin
from .integrations import PeftAdapterMixin, deepspeed_config, is_deepspeed_zero3_enabled
from .pytorch_utils import (  # noqa: F401
from .quantizers import AutoHfQuantizer, HfQuantizer
from .safetensors_conversion import auto_conversion
from .utils import (
from .utils.hub import convert_file_size_to_int, create_and_tag_model_card, get_checkpoint_shard_files
from .utils.import_utils import (
from .utils.quantization_config import BitsAndBytesConfig, QuantizationMethod
def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
    """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".

        We pass the `__call__` method of the modules instead of `forward` because `__call__` attaches all the hooks of
        the module. https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2

        Args:
            gradient_checkpointing_kwargs (dict, *optional*):
                Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
        """
    if not self.supports_gradient_checkpointing:
        raise ValueError(f'{self.__class__.__name__} does not support gradient checkpointing.')
    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {'use_reentrant': True}
    gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)
    _is_using_old_format = 'value' in inspect.signature(self._set_gradient_checkpointing).parameters
    if not _is_using_old_format:
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
    else:
        self.apply(partial(self._set_gradient_checkpointing, value=True))
        logger.warn('You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it).Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model.')
    if getattr(self, '_hf_peft_config_loaded', False):
        self.enable_input_require_grads()