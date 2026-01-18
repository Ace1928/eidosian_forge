from collections import namedtuple
from typing import Optional, Any, Union, Type
import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import (
from .observer import (
import warnings
import copy
def get_default_qat_qconfig_dict(backend='x86', version=1):
    warnings.warn('torch.ao.quantization.get_default_qat_qconfig_dict is deprecated and will be removed in a future version. Please use torch.ao.quantization.get_default_qat_qconfig_mapping instead.')
    return torch.ao.quantization.get_default_qat_qconfig_mapping(backend, version).to_dict()