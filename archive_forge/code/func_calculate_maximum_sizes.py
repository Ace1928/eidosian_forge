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
def calculate_maximum_sizes(model: torch.nn.Module):
    """Computes the total size of the model and its largest layer"""
    sizes = compute_module_sizes(model)
    no_split_modules = getattr(model, '_no_split_modules', None)
    if no_split_modules is None:
        no_split_modules = []
    modules_to_treat = list(model.named_parameters(recurse=False)) + list(model.named_children()) + list(model.named_buffers(recurse=False))
    largest_layer = get_max_layer_size(modules_to_treat, sizes, no_split_modules)
    total_size = sizes['']
    return (total_size, largest_layer)