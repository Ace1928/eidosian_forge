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
def enable_adapters(self, enabled: bool) -> None:
    """Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
    if enabled:
        self.set_adapter(self.active_adapters)
        self._disable_adapters = False
    else:
        for layer_name in self.adapter_layer_names:
            layer = getattr(self, layer_name)
            layer.requires_grad_(False)
        self._disable_adapters = True