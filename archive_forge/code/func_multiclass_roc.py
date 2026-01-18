from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.compute import _safe_divide, interp
from torchmetrics.utilities.enums import ClassificationTask
def multiclass_roc(preds: Tensor, target: Tensor, num_classes: int, thresholds: Optional[Union[int, List[float], Tensor]]=None, average: Optional[Literal['micro', 'macro']]=None, ignore_index: Optional[int]=None, validate_args: bool=True) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    """Compute the Receiver Operating Characteristic (ROC) for multiclass tasks.

    The curve consist of multiple pairs of true positive rate (TPR) and false positive rate (FPR) values evaluated at
    different thresholds, such that the tradeoff between the two values can be seen.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      softmax per sample.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds} \\times n_{classes})` (constant memory).

    Note that outputted thresholds will be in reversed order to ensure that they corresponds to both fpr and tpr which
    are sorted in reversed order during their calculation, such that they are monotome increasing.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifying the number of classes
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        average:
            If aggregation of curves should be applied. By default, the curves are not aggregated and a curve for
            each class is returned. If `average` is set to ``"micro"``, the metric will aggregate the curves by one hot
            encoding the targets and flattening the predictions, considering all classes jointly as a binary problem.
            If `average` is set to ``"macro"``, the metric will aggregate the curves by first interpolating the curves
            from each class at a combined set of thresholds and then average over the classwise interpolated curves.
            See `averaging curve objects`_ for more info on the different averaging methods.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        (tuple): a tuple of either 3 tensors or 3 lists containing

        - fpr: if `thresholds=None` a list for each class is returned with an 1d tensor of size (n_thresholds+1, )
          with false positive rate values (length may differ between classes). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_classes, n_thresholds+1) with false positive rate values is returned.
        - tpr: if `thresholds=None` a list for each class is returned with an 1d tensor of size (n_thresholds+1, )
          with true positive rate values (length may differ between classes). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_classes, n_thresholds+1) with true positive rate values is returned.
        - thresholds: if `thresholds=None` a list for each class is returned with an 1d tensor of size (n_thresholds, )
          with decreasing threshold values (length may differ between classes). If `threshold` is set to something else,
          then a single 1d tensor of size (n_thresholds, ) is returned with shared threshold values for all classes.

    Example:
        >>> from torchmetrics.functional.classification import multiclass_roc
        >>> preds = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> fpr, tpr, thresholds = multiclass_roc(
        ...    preds, target, num_classes=5, thresholds=None
        ... )
        >>> fpr  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0., 0., 1.]), tensor([0., 0., 1.]), tensor([0.0000, 0.3333, 1.0000]),
         tensor([0.0000, 0.3333, 1.0000]), tensor([0., 1.])]
        >>> tpr
        [tensor([0., 1., 1.]), tensor([0., 1., 1.]), tensor([0., 0., 1.]), tensor([0., 0., 1.]), tensor([0., 0.])]
        >>> thresholds  # doctest: +NORMALIZE_WHITESPACE
        [tensor([1.0000, 0.7500, 0.0500]), tensor([1.0000, 0.7500, 0.0500]),
         tensor([1.0000, 0.7500, 0.0500]), tensor([1.0000, 0.7500, 0.0500]), tensor([1.0000, 0.0500])]
        >>> multiclass_roc(
        ...     preds, target, num_classes=5, thresholds=5
        ... )  # doctest: +NORMALIZE_WHITESPACE
        (tensor([[0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
                 [0.0000, 0.3333, 0.3333, 0.3333, 1.0000],
                 [0.0000, 0.3333, 0.3333, 0.3333, 1.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 1.0000]]),
         tensor([[0., 1., 1., 1., 1.],
                 [0., 1., 1., 1., 1.],
                 [0., 0., 0., 0., 1.],
                 [0., 0., 0., 0., 1.],
                 [0., 0., 0., 0., 0.]]),
         tensor([1.0000, 0.7500, 0.5000, 0.2500, 0.0000]))

    """
    if validate_args:
        _multiclass_precision_recall_curve_arg_validation(num_classes, thresholds, ignore_index, average)
        _multiclass_precision_recall_curve_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target, thresholds = _multiclass_precision_recall_curve_format(preds, target, num_classes, thresholds, ignore_index, average)
    state = _multiclass_precision_recall_curve_update(preds, target, num_classes, thresholds, average)
    return _multiclass_roc_compute(state, num_classes, thresholds, average)