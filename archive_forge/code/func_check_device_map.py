import contextlib
import gc
import importlib
import inspect
import json
import logging
import os
import re
import shutil
import tempfile
import warnings
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union
import packaging
import torch
import torch.nn as nn
from ..state import AcceleratorState
from .constants import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from .dataclasses import AutocastKwargs, CustomDtype, DistributedType
from .imports import is_mps_available, is_npu_available, is_peft_available, is_torch_xla_available, is_xpu_available
from .offload import load_offloaded_weight, offload_weight, save_offload_index
from .tqdm import is_tqdm_available, tqdm
from .versions import compare_versions
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
def check_device_map(model: nn.Module, device_map: Dict[str, Union[int, str, torch.device]]):
    """
    Checks a device map covers everything in a given model.

    Args:
        model (`torch.nn.Module`): The model to check the device map against.
        device_map (`Dict[str, Union[int, str, torch.device]]`): The device map to check.
    """
    all_model_tensors = [name for name, _ in model.state_dict().items()]
    for module_name in device_map.keys():
        if module_name == '':
            all_model_tensors.clear()
            break
        else:
            all_model_tensors = [name for name in all_model_tensors if not name == module_name and (not name.startswith(module_name + '.'))]
    if len(all_model_tensors) > 0:
        non_covered_params = ', '.join(all_model_tensors)
        raise ValueError(f'The device_map provided does not give any device for the following parameters: {non_covered_params}')