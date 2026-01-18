from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.quant_type import QuantType, _quant_type_from_str, _get_quant_type_to_str
def set_float_to_observed_mapping(self, float_class: Type, observed_class: Type, quant_type: QuantType=QuantType.STATIC) -> PrepareCustomConfig:
    """
        Set the mapping from a custom float module class to a custom observed module class.

        The observed module class must have a ``from_float`` class method that converts the float module class
        to the observed module class. This is currently only supported for static quantization.
        """
    if quant_type != QuantType.STATIC:
        raise ValueError('set_float_to_observed_mapping is currently only supported for static quantization')
    if quant_type not in self.float_to_observed_mapping:
        self.float_to_observed_mapping[quant_type] = {}
    self.float_to_observed_mapping[quant_type][float_class] = observed_class
    return self