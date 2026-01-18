from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import ModuleList
from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_single_or_multi_val
from torchmetrics.utilities.prints import rank_zero_warn
@property
def n_steps(self) -> int:
    """Returns the number of times the tracker has been incremented."""
    return len(self) - 1