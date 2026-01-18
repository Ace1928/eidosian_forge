from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount, _cumsum, dim_zero_cat
from torchmetrics.utilities.enums import EnumStr
def kendall_rank_corrcoef(preds: Tensor, target: Tensor, variant: Literal['a', 'b', 'c']='b', t_test: bool=False, alternative: Optional[Literal['two-sided', 'less', 'greater']]='two-sided') -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Compute `Kendall Rank Correlation Coefficient`_.

    .. math::
        tau_a = \\frac{C - D}{C + D}

    where :math:`C` represents concordant pairs, :math:`D` stands for discordant pairs.

    .. math::
        tau_b = \\frac{C - D}{\\sqrt{(C + D + T_{preds}) * (C + D + T_{target})}}

    where :math:`C` represents concordant pairs, :math:`D` stands for discordant pairs and :math:`T` represents
    a total number of ties.

    .. math::
        tau_c = 2 * \\frac{C - D}{n^2 * \\frac{m - 1}{m}}

    where :math:`C` represents concordant pairs, :math:`D` stands for discordant pairs, :math:`n` is a total number
    of observations and :math:`m` is a ``min`` of unique values in ``preds`` and ``target`` sequence.

    Definitions according to Definition according to `The Treatment of Ties in Ranking Problems`_.

    Args:
        preds: Sequence of data of either shape ``(N,)`` or ``(N,d)``
        target: Sequence of data of either shape ``(N,)`` or ``(N,d)``
        variant: Indication of which variant of Kendall's tau to be used
        t_test: Indication whether to run t-test
        alternative: Alternative hypothesis for t-test. Possible values:
            - 'two-sided': the rank correlation is nonzero
            - 'less': the rank correlation is negative (less than zero)
            - 'greater':  the rank correlation is positive (greater than zero)

    Return:
        Correlation tau statistic
        (Optional) p-value of corresponding statistical test (asymptotic)

    Raises:
        ValueError: If ``t_test`` is not of a type bool
        ValueError: If ``t_test=True`` and ``alternative=None``

    Example (single output regression):
        >>> from torchmetrics.functional.regression import kendall_rank_corrcoef
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> target = torch.tensor([3, -0.5, 2, 1])
        >>> kendall_rank_corrcoef(preds, target)
        tensor(0.3333)

    Example (multi output regression):
        >>> from torchmetrics.functional.regression import kendall_rank_corrcoef
        >>> preds = torch.tensor([[2.5, 0.0], [2, 8]])
        >>> target = torch.tensor([[3, -0.5], [2, 1]])
        >>> kendall_rank_corrcoef(preds, target)
        tensor([1., 1.])

    Example (single output regression with t-test)
        >>> from torchmetrics.functional.regression import kendall_rank_corrcoef
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> target = torch.tensor([3, -0.5, 2, 1])
        >>> kendall_rank_corrcoef(preds, target, t_test=True, alternative='two-sided')
        (tensor(0.3333), tensor(0.4969))

    Example (multi output regression with t-test):
        >>> from torchmetrics.functional.regression import kendall_rank_corrcoef
        >>> preds = torch.tensor([[2.5, 0.0], [2, 8]])
        >>> target = torch.tensor([[3, -0.5], [2, 1]])
        >>> kendall_rank_corrcoef(preds, target, t_test=True, alternative='two-sided')
            (tensor([1., 1.]), tensor([nan, nan]))

    """
    if not isinstance(t_test, bool):
        raise ValueError(f'Argument `t_test` is expected to be of a type `bool`, but got {type(t_test)}.')
    if t_test and alternative is None:
        raise ValueError('Argument `alternative` is required if `t_test=True` but got `None`.')
    _variant = _MetricVariant.from_str(str(variant))
    _alternative = _TestAlternative.from_str(str(alternative)) if t_test else None
    _preds, _target = _kendall_corrcoef_update(preds, target, [], [], num_outputs=1 if preds.ndim == 1 else preds.shape[-1])
    tau, p_value = _kendall_corrcoef_compute(dim_zero_cat(_preds), dim_zero_cat(_target), _variant, _alternative)
    if p_value is not None:
        return (tau, p_value)
    return tau