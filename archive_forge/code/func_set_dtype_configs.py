from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum
def set_dtype_configs(self, dtype_configs: List[DTypeConfig]) -> BackendPatternConfig:
    """
        Set the supported data types passed as arguments to quantize ops in the
        reference model spec, overriding all previously registered data types.
        """
    self.dtype_configs = dtype_configs
    return self