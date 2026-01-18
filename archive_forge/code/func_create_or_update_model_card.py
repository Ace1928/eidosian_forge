from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Any, Optional, Union
import torch
from accelerate.hooks import remove_hook_from_submodules
from torch import nn
from transformers.utils import PushToHubMixin
from peft.tuners.mixed import COMPATIBLE_TUNER_TYPES
from .config import PeftConfig
from .peft_model import PeftModel
from .tuners import (
from .utils import PeftType, _set_adapter, _set_trainable
def create_or_update_model_card(self, output_dir: str):
    raise NotImplementedError(f'Model card creation is not supported for {self.__class__.__name__} (yet).')