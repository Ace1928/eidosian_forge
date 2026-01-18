import sys
from fractions import Fraction
from math import ceil
from typing import cast, List, Optional, Sequence
def ratio_distribute(total: int, ratios: List[int], minimums: Optional[List[int]]=None) -> List[int]:
    """Distribute an integer total in to parts based on ratios.

    Args:
        total (int): The total to divide.
        ratios (List[int]): A list of integer ratios.
        minimums (List[int]): List of minimum values for each slot.

    Returns:
        List[int]: A list of integers guaranteed to sum to total.
    """
    if minimums:
        ratios = [ratio if _min else 0 for ratio, _min in zip(ratios, minimums)]
    total_ratio = sum(ratios)
    assert total_ratio > 0, 'Sum of ratios must be > 0'
    total_remaining = total
    distributed_total: List[int] = []
    append = distributed_total.append
    if minimums is None:
        _minimums = [0] * len(ratios)
    else:
        _minimums = minimums
    for ratio, minimum in zip(ratios, _minimums):
        if total_ratio > 0:
            distributed = max(minimum, ceil(ratio * total_remaining / total_ratio))
        else:
            distributed = total_remaining
        append(distributed)
        total_ratio -= ratio
        total_remaining -= distributed
    return distributed_total