from typing import Optional
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities.compute import _adjust_weights_safe_divide, _safe_divide
from torchmetrics.utilities.enums import ClassificationTask
def multiclass_f1_score(preds: Tensor, target: Tensor, num_classes: int, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]='macro', top_k: int=1, multidim_average: Literal['global', 'samplewise']='global', ignore_index: Optional[int]=None, validate_args: bool=True) -> Tensor:
    """Compute F-1 score for multiclass tasks.

    .. math::
        F_{1} = 2\\frac{\\text{precision} * \\text{recall}}{(\\text{precision}) + \\text{recall}}

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``torch.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifying the number of classes
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction
        top_k:
            Number of highest probability or logit score predictions considered to find the correct label.
            Only works when ``preds`` contain probabilities/logits.
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The returned shape depends on the ``average`` and ``multidim_average`` arguments:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
          - If ``average=None/'none'``, the shape will be ``(C,)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
          - If ``average=None/'none'``, the shape will be ``(N, C)``

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multiclass_f1_score
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> multiclass_f1_score(preds, target, num_classes=3)
        tensor(0.7778)
        >>> multiclass_f1_score(preds, target, num_classes=3, average=None)
        tensor([0.6667, 0.6667, 1.0000])

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import multiclass_f1_score
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58],
        ...                 [0.22, 0.61, 0.17],
        ...                 [0.71, 0.09, 0.20],
        ...                 [0.05, 0.82, 0.13]])
        >>> multiclass_f1_score(preds, target, num_classes=3)
        tensor(0.7778)
        >>> multiclass_f1_score(preds, target, num_classes=3, average=None)
        tensor([0.6667, 0.6667, 1.0000])

    Example (multidim tensors):
        >>> from torchmetrics.functional.classification import multiclass_f1_score
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
        >>> multiclass_f1_score(preds, target, num_classes=3, multidim_average='samplewise')
        tensor([0.4333, 0.2667])
        >>> multiclass_f1_score(preds, target, num_classes=3, multidim_average='samplewise', average=None)
        tensor([[0.8000, 0.0000, 0.5000],
                [0.0000, 0.4000, 0.4000]])

    """
    return multiclass_fbeta_score(preds=preds, target=target, beta=1.0, num_classes=num_classes, average=average, top_k=top_k, multidim_average=multidim_average, ignore_index=ignore_index, validate_args=validate_args)