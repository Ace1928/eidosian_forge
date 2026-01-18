from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
def calibration_error(preds: Tensor, target: Tensor, task: Literal['binary', 'multiclass'], n_bins: int=15, norm: Literal['l1', 'l2', 'max']='l1', num_classes: Optional[int]=None, ignore_index: Optional[int]=None, validate_args: bool=True) -> Tensor:
    """`Top-label Calibration Error`_.

    The expected calibration error can be used to quantify how well a given model is calibrated e.g. how well the
    predicted output probabilities of the model matches the actual probabilities of the ground truth distribution.
    Three different norms are implemented, each corresponding to variations on the calibration error metric.

    .. math::
        \\text{ECE} = \\sum_i^N b_i \\|(p_i - c_i)\\|, \\text{L1 norm (Expected Calibration Error)}

    .. math::
        \\text{MCE} =  \\max_{i} (p_i - c_i), \\text{Infinity norm (Maximum Calibration Error)}

    .. math::
        \\text{RMSCE} = \\sqrt{\\sum_i^N b_i(p_i - c_i)^2}, \\text{L2 norm (Root Mean Square Calibration Error)}

    Where :math:`p_i` is the top-1 prediction accuracy in bin :math:`i`, :math:`c_i` is the average confidence of
    predictions in bin :math:`i`, and :math:`b_i` is the fraction of data points in bin :math:`i`. Bins are constructed
    in an uniform way in the [0,1] range.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'`` or ``'multiclass'``. See the documentation of
    :func:`~torchmetrics.functional.classification.binary_calibration_error` and
    :func:`~torchmetrics.functional.classification.multiclass_calibration_error` for the specific details of
    each argument influence and examples.

    """
    task = ClassificationTaskNoMultilabel.from_str(task)
    assert norm is not None
    if task == ClassificationTaskNoMultilabel.BINARY:
        return binary_calibration_error(preds, target, n_bins, norm, ignore_index, validate_args)
    if task == ClassificationTaskNoMultilabel.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f'`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`')
        return multiclass_calibration_error(preds, target, num_classes, n_bins, norm, ignore_index, validate_args)
    raise ValueError(f"Expected argument `task` to either be `'binary'` or `'multiclass'` but got {task}")