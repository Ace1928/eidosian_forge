from __future__ import annotations
import logging
import re
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Optional, Union
import torch
from accelerate.hooks import AlignDevicesHook
from accelerate.utils import named_module_tensors, offload_state_dict
from torch import nn
from transformers import PreTrainedModel
from transformers.pytorch_utils import Conv1D
from peft.utils import INCLUDE_LINEAR_LAYERS_SHORTHAND
from ..config import PeftConfig
from ..utils import ModulesToSaveWrapper, _get_submodules
def get_base_layer(self) -> nn.Module:
    """
        (Recursively) get the base_layer.

        This is necessary for the case that the tuner layer wraps another tuner layer.

        """
    base_layer = self
    while hasattr(base_layer, 'base_layer'):
        base_layer = base_layer.base_layer
    return base_layer