import math
from typing import Literal, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from torchmetrics.functional.image.lpips import _LPIPS
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE
@property
def num_classes(self) -> int:
    """Return the number of classes for conditional generation."""
    raise NotImplementedError