from typing import Any, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics.image.fid import NoTrainInceptionV3
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TORCH_FIDELITY_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def poly_kernel(f1: Tensor, f2: Tensor, degree: int=3, gamma: Optional[float]=None, coef: float=1.0) -> Tensor:
    """Adapted from `KID Score`_."""
    if gamma is None:
        gamma = 1.0 / f1.shape[1]
    return (f1 @ f2.T * gamma + coef) ** degree