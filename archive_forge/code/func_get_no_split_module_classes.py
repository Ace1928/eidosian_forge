import json
import os
from enum import Enum
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.pytorch_utils import Conv1D
from transformers.utils.quantization_config import QuantizationMethod
from ..utils import is_accelerate_available, is_auto_gptq_available
from ..utils.modeling_utils import recurse_getattr
from .constants import GPTQ_CONFIG
from .data import get_dataset, prepare_dataset
from .utils import get_block_name_with_pattern, get_device, get_layers, get_preceding_modules, get_seqlen
def get_no_split_module_classes(self, model):
    """
        Get the modules that should not be split across multiple devices.
        Args:
            model (`nn.Module`):
                The input model
        """
    block_class_name = recurse_getattr(model, self.block_name_to_quantize)[0].__class__.__name__
    no_split_module_classes = [block_class_name]
    return no_split_module_classes