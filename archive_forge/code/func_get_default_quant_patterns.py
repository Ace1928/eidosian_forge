from collections import OrderedDict
from typing import Dict, Any
from torch.ao.quantization.utils import Pattern
from ..fake_quantize import FixedQParamsFakeQuantize
from ..observer import ObserverBase
import copy
def get_default_quant_patterns() -> Dict[Pattern, QuantizeHandler]:
    return copy.copy(_DEFAULT_QUANTIZATION_PATTERNS)