from typing import Dict, Any, List, Callable, Union, Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backend_config import (
from ..utils import Pattern
from ..fuser_method_mappings import (
def get_fused_module_classes(backend_config: BackendConfig) -> Tuple[type, ...]:
    fused_module_classes = []
    for config in backend_config.configs:
        if config.fused_module is not None:
            fused_module_classes.append(config.fused_module)
    return tuple(set(fused_module_classes))