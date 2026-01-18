from typing import Dict, Any, List, Callable, Union, Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backend_config import (
from ..utils import Pattern
from ..fuser_method_mappings import (
def get_pattern_to_dtype_configs(backend_config: BackendConfig) -> Dict[Pattern, List[DTypeConfig]]:
    pattern_to_dtype_configs: Dict[Pattern, List[DTypeConfig]] = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        pattern_to_dtype_configs[pattern] = config.dtype_configs
    return pattern_to_dtype_configs