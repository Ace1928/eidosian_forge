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
def retie_parameters(model, tied_params):
    """
    Reties tied parameters in a given model if the link was broken (for instance when adding hooks).

    Args:
        model (`torch.nn.Module`):
            The model in which to retie parameters.
        tied_params (`List[List[str]]`):
            A mapping parameter name to tied parameter name as obtained by `find_tied_parameters`.
    """
    for tied_group in tied_params:
        param_to_tie = None
        for param_name in tied_group:
            module = model
            splits = param_name.split('.')
            for split in splits[:-1]:
                module = getattr(module, split)
            param = getattr(module, splits[-1])
            if param_to_tie is None and param.device != torch.device('meta'):
                param_to_tie = param
                break
        if param_to_tie is not None:
            for param_name in tied_group:
                module = model
                splits = param_name.split('.')
                for split in splits[:-1]:
                    module = getattr(module, split)
                setattr(module, splits[-1], param_to_tie)