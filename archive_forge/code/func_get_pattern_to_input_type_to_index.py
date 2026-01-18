from typing import Dict, Any, List, Callable, Union, Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backend_config import (
from ..utils import Pattern
from ..fuser_method_mappings import (
def get_pattern_to_input_type_to_index(backend_config: BackendConfig) -> Dict[Pattern, Dict[str, int]]:
    pattern_to_input_type_to_index: Dict[Pattern, Dict[str, int]] = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        pattern_to_input_type_to_index[pattern] = config._input_type_to_index
    return pattern_to_input_type_to_index