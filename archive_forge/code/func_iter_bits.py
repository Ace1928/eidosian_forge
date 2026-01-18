from typing import Iterator, Tuple
import numpy as np
def iter_bits(val: int, width: int, *, signed: bool=False) -> Iterator[int]:
    """Iterate over the bits in a binary representation of `val`.

    This uses a big-endian convention where the most significant bit
    is yielded first.

    Args:
        val: The integer value. Its bitsize must fit within `width`
        width: The number of output bits.
        signed: If True, the most significant bit represents the sign of
            the number (ones complement) which is 1 if val < 0 else 0.
    Raises:
        ValueError: If `val` is negative or if `val.bit_length()` exceeds `width`.
    """
    if val.bit_length() + int(val < 0) > width:
        raise ValueError(f'{val} exceeds width {width}.')
    if val < 0 and (not signed):
        raise ValueError(f'{val} is negative.')
    if signed:
        yield (1 if val < 0 else 0)
        width -= 1
    for b in f'{abs(val):0{width}b}':
        yield int(b)