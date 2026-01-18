from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.quant_type import QuantType, _quant_type_from_str, _get_quant_type_to_str
def set_preserved_attributes(self, attributes: List[str]) -> FuseCustomConfig:
    """
        Set the names of the attributes that will persist in the graph module even if they are not used in
        the model's ``forward`` method.
        """
    self.preserved_attributes = attributes
    return self