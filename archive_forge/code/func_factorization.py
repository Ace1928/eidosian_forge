import math
from typing import Any, Optional, Set, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lycoris_utils import LycorisLayer
def factorization(dimension: int, factor: int=-1) -> Tuple[int, int]:
    """Factorizes the provided number into the product of two numbers

    Args:
        dimension (`int`): The number that needs to be factorized.
        factor (`int`, optional):
            Factorization divider. The algorithm will try to output two numbers, one of each will be as close to the
            factor as possible. If -1 is provided, the decomposition algorithm would try to search dividers near the
            square root of the dimension. Defaults to -1.

    Returns:
        Tuple[`int`, `int`]: A tuple of two numbers, whose product is equal to the provided number. The first number is
        always less than or equal to the second.

    Example:
        ```py
        >>> factorization(256, factor=-1)
        (16, 16)

        >>> factorization(128, factor=-1)
        (8, 16)

        >>> factorization(127, factor=-1)
        (1, 127)

        >>> factorization(128, factor=4)
        (4, 32)
        ```
    """
    if factor > 0 and dimension % factor == 0:
        m = factor
        n = dimension // factor
        return (m, n)
    if factor == -1:
        factor = dimension
    m, n = (1, dimension)
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = (new_m, new_n)
    if m > n:
        n, m = (m, n)
    return (m, n)