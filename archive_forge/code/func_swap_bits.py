from __future__ import annotations
from typing import Union
import numpy as np
def swap_bits(num: int, a: int, b: int) -> int:
    """
    Swaps the bits at positions 'a' and 'b' in the number 'num'.

    Args:
        num: an integer number where bits should be swapped.
        a: index of the first bit to be swapped.
        b: index of the second bit to be swapped.

    Returns:
        the number with swapped bits.
    """
    x = (num >> a ^ num >> b) & 1
    return num ^ (x << a | x << b)