from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.quant_type import QuantType, _quant_type_from_str, _get_quant_type_to_str
def set_observed_to_quantized_mapping(self, observed_class: Type, quantized_class: Type, quant_type: QuantType=QuantType.STATIC) -> ConvertCustomConfig:
    """
        Set the mapping from a custom observed module class to a custom quantized module class.

        The quantized module class must have a ``from_observed`` class method that converts the observed module class
        to the quantized module class.
        """
    if quant_type not in self.observed_to_quantized_mapping:
        self.observed_to_quantized_mapping[quant_type] = {}
    self.observed_to_quantized_mapping[quant_type][observed_class] = quantized_class
    return self