from __future__ import annotations
from typing import Union
import numpy as np
def reverse_bits(x: Union[int, np.ndarray], nbits: int, enable: bool) -> Union[int, np.ndarray]:
    """
    Reverses the bit order in a number of ``nbits`` length.
    If ``x`` is an array, then operation is applied to every entry.

    Args:
        x: either a single integer or an array of integers.
        nbits: number of meaningful bits in the number x.
        enable: apply reverse operation, if enabled, otherwise leave unchanged.

    Returns:
        a number or array of numbers with reversed bits.
    """
    if not enable:
        if isinstance(x, int):
            pass
        else:
            x = x.copy()
        return x
    if isinstance(x, int):
        res: int | np.ndarray = int(0)
    else:
        x = x.copy()
        res = np.full_like(x, fill_value=0)
    for _ in range(nbits):
        res <<= 1
        res |= x & 1
        x >>= 1
    return res