import torch
from torch.nn import Module
from torch.ao.quantization.observer import (
import re
from abc import ABC, abstractmethod
from typing import Any, Tuple
@torch.jit.export
def enable_fake_quant(self, enabled: bool=True) -> None:
    self.fake_quant_enabled[0] = 1 if enabled else 0