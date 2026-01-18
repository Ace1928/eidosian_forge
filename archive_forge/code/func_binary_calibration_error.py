from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
def binary_calibration_error(preds: Tensor, target: Tensor, n_bins: int=15, norm: Literal['l1', 'l2', 'max']='l1', ignore_index: Optional[int]=None, validate_args: bool=True) -> Tensor:
    """`Top-label Calibration Error`_ for binary tasks.

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

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified). The value 1 always encodes the positive class.

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        n_bins: Number of bins to use when computing the metric.
        norm: Norm used to compare empirical and expected probability bins.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example:
        >>> from torchmetrics.functional.classification import binary_calibration_error
        >>> preds = torch.tensor([0.25, 0.25, 0.55, 0.75, 0.75])
        >>> target = torch.tensor([0, 0, 1, 1, 1])
        >>> binary_calibration_error(preds, target, n_bins=2, norm='l1')
        tensor(0.2900)
        >>> binary_calibration_error(preds, target, n_bins=2, norm='l2')
        tensor(0.2918)
        >>> binary_calibration_error(preds, target, n_bins=2, norm='max')
        tensor(0.3167)

    """
    if validate_args:
        _binary_calibration_error_arg_validation(n_bins, norm, ignore_index)
        _binary_calibration_error_tensor_validation(preds, target, ignore_index)
    preds, target = _binary_confusion_matrix_format(preds, target, threshold=0.0, ignore_index=ignore_index, convert_to_labels=False)
    confidences, accuracies = _binary_calibration_error_update(preds, target)
    return _ce_compute(confidences, accuracies, n_bins, norm)