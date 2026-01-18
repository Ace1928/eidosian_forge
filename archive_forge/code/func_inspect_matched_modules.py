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
def inspect_matched_modules(tuner: BaseTuner, adapter_name: str='default') -> dict:
    """
    A helper function to inspect the set of matched and unmatched modules for a PEFT model and the given adapter.
    """
    config = tuner.peft_config[adapter_name]
    key_list = [key for key, _ in tuner.model.named_modules()]
    module_dict = {'matched': [], 'unmatched': []}
    for key in key_list:
        if tuner._check_target_module_exists(config, key):
            module_dict['matched'].append(key)
        else:
            module_dict['unmatched'].append(key)
    return module_dict