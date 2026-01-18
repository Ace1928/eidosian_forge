from typing import TYPE_CHECKING, Any, ContextManager, Dict, Literal, Optional, cast
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import get_args, override
from lightning_fabric.plugins.precision.amp import _optimizer_handles_unscaling
from lightning_fabric.plugins.precision.precision import Precision
from lightning_fabric.plugins.precision.utils import _convert_fp_tensor, _DtypeContextManager
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.types import Optimizable
@override
def module_init_context(self) -> ContextManager:
    return _DtypeContextManager(self.mixed_precision_config.param_dtype or torch.float32)