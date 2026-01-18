from typing import Optional, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
def check_cluster_labels(preds: Tensor, target: Tensor) -> None:
    """Check shape of input tensors and if they are real, discrete tensors.

    Args:
        preds: predicted labels
        target: ground truth labels

    """
    _check_same_shape(preds, target)
    if not (_is_real_discrete_label(preds) and _is_real_discrete_label(target)):
        raise ValueError(f'Expected real, discrete values for x but received {preds.dtype} and {target.dtype}.')