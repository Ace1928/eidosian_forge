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
def check_tied_parameters_in_config(model: nn.Module):
    """
    Check if there is any indication in the given model that some weights should be tied.

    Args:
        model (`torch.nn.Module`): The model to inspect

    Returns:
        bool: True if the model needs to have tied weights
    """
    has_tied_word_embedding = False
    has_tied_encoder_decoder = False
    has_tied_module = False
    if 'PreTrainedModel' in [c.__name__ for c in inspect.getmro(model.__class__)]:
        has_tied_word_embedding = hasattr(model, 'config') and getattr(model.config, 'tie_word_embeddings', False) and model.get_output_embeddings()
        has_tied_encoder_decoder = hasattr(model, 'config') and getattr(model.config, 'is_encoder_decoder', False) and getattr(model.config, 'tie_encoder_decoder', False)
        has_tied_module = any((hasattr(module, '_tie_weights') for module in model.modules()))
    return any([has_tied_word_embedding, has_tied_encoder_decoder, has_tied_module])