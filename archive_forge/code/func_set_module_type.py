from __future__ import annotations
import copy
import functools
from typing import Any, Callable, Dict, List, Optional, Set
import torch
import torch._dynamo as torchdynamo
import torch.nn.functional as F
from torch.ao.quantization.fake_quantize import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer import QuantizationSpec, Quantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
from torch.fx import Node
def set_module_type(self, module_type: Callable, quantization_config: QuantizationConfig):
    """Set quantization_config for a submodule with type: `module_type`, for example:
        quantizer.set_module_name(Sub) or quantizer.set_module_name(nn.Linear), it will quantize all supported operator/operator
        patterns in the submodule with this module type with the given `quantization_config`
        """
    self.module_type_config[module_type] = quantization_config
    return self