import copy
import torch
from torch import nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.reference as nnqr
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.qat as nnqat
import torch.ao.nn.qat.dynamic as nnqatd
from typing import Optional, Union, Dict, Set, Callable, Any
import torch.ao.nn.sparse
import torch.ao.nn as ao_nn
from torch.ao.quantization.stubs import QuantStub, DeQuantStub
from torch.ao.quantization.fake_quantize import (
from torch.ao.quantization.utils import get_combined_dict
from torch.nn.utils.parametrize import type_before_parametrizations
def get_default_static_quant_reference_module_mappings() -> Dict[Callable, Any]:
    """ Get reference module mapping for post training static quantization
    """
    return copy.deepcopy(DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS)