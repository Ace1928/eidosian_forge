from typing import Dict, Any, List, Callable, Union, Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backend_config import (
from ..utils import Pattern
from ..fuser_method_mappings import (
def get_qat_module_classes(backend_config: BackendConfig) -> Tuple[type, ...]:
    qat_module_classes = []
    for config in backend_config.configs:
        if config.qat_module is not None:
            qat_module_classes.append(config.qat_module)
    return tuple(set(qat_module_classes))