from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.quant_type import QuantType, _quant_type_from_str, _get_quant_type_to_str
def set_output_quantized_indexes(self, indexes: List[int]) -> PrepareCustomConfig:
    """
        Set the indexes of the outputs of the graph that should be quantized.
        Outputs are otherwise assumed to be in fp32 by default instead.
        """
    self.output_quantized_indexes = indexes
    return self