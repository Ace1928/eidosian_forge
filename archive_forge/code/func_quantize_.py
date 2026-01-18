import functools
import logging
import math
import os
import warnings
from contextlib import ExitStack
from functools import partial
from types import ModuleType
from typing import Any, Callable, ContextManager, Literal, Optional, OrderedDict, Set, Tuple, Type, cast
import torch
from lightning_utilities import apply_to_collection
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.nn import init
from torch.nn.modules.module import _IncompatibleKeys
from typing_extensions import Self, override
from lightning_fabric.plugins.precision.precision import Precision
from lightning_fabric.plugins.precision.utils import (
from lightning_fabric.utilities.types import _DEVICE
def quantize_(self, weight: Optional[torch.Tensor]=None, device: Optional[torch.device]=None) -> None:
    """Inplace quantize."""
    if weight is None:
        weight = self.weight.data
        if weight.data.type == torch.uint8:
            return
    assert isinstance(self.weight, bnb.nn.Params4bit)
    self.weight = self.quantize(self.weight, weight, device)