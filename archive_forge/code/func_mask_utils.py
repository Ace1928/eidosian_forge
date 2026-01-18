import contextlib
import io
import json
from types import ModuleType
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch import distributed as dist
from typing_extensions import Literal
from torchmetrics.detection.helpers import _fix_empty_tensors, _input_validator, _validate_iou_type_arg
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import (
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
@property
def mask_utils(self) -> object:
    """Returns the mask utils object for the given backend, done in this way to make metric picklable."""
    _, _, mask_utils = _load_backend_tools(self.backend)
    return mask_utils