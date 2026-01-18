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
def load_offloaded_weights(model, index, offload_folder):
    """
    Loads the weights from the offload folder into the model.

    Args:
        model (`torch.nn.Module`):
            The model to load the weights into.
        index (`dict`):
            A dictionary containing the parameter name and its metadata for each parameter that was offloaded from the
            model.
        offload_folder (`str`):
            The folder where the offloaded weights are stored.
    """
    if index is None or len(index) == 0:
        return
    for param_name, metadata in index.items():
        if 'SCB' in param_name:
            continue
        fp16_statistics = None
        if 'weight' in param_name and param_name.replace('weight', 'SCB') in index.keys():
            weight_name = param_name.replace('weight', 'SCB')
            fp16_statistics = load_offloaded_weight(os.path.join(offload_folder, f'{weight_name}.dat'), index[weight_name])
        tensor_file = os.path.join(offload_folder, f'{param_name}.dat')
        weight = load_offloaded_weight(tensor_file, metadata)
        set_module_tensor_to_device(model, param_name, 'cpu', value=weight, fp16_statistics=fp16_statistics)