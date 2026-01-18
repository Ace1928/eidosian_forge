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
def compute_module_total_buffer_size(model: nn.Module, dtype: Optional[Union[str, torch.device]]=None, special_dtypes: Optional[Dict[str, Union[str, torch.device]]]=None):
    """
    Compute the total size of buffers in each submodule of a given model.
    """
    module_sizes = compute_module_sizes(model, dtype=dtype, special_dtypes=special_dtypes, buffers_only=True)
    return module_sizes.get('', 0)