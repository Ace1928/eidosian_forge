from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.quant_type import QuantType, _quant_type_from_str, _get_quant_type_to_str
def set_non_traceable_module_classes(self, module_classes: List[Type]) -> PrepareCustomConfig:
    """
        Set the modules that are not symbolically traceable, identified by class.
        """
    self.non_traceable_module_classes = module_classes
    return self