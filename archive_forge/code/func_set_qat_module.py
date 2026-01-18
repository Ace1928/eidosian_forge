from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum
def set_qat_module(self, qat_module: Type[torch.nn.Module]) -> BackendPatternConfig:
    """
        Set the module that represents the QAT implementation for this pattern.
        """
    self.qat_module = qat_module
    return self