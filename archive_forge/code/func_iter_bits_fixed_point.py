from typing import Iterator, Tuple
import numpy as np
def iter_bits_fixed_point(val: float, width: int, *, signed: bool=False) -> Iterator[int]:
    """Represent the floating point number -1 <= val <= 1 using `width` bits.

    $$
        val = \\sum_{b=0}^{width - 1} val[b] / 2^{1+b}
    $$

    Args:
        val: Floating point number in [-1, 1]
        width: The number of output bits in fixed point binary representation of `val`.
        signed: If True, the most significant bit represents the sign of
            the number (ones complement) which is 1 if val < 0 else 0.

    Raises:
        ValueError: If val is not between [0, 1] (signed=False) / [-1, 1] (signed=True).
    """
    lb = -1 if signed else 0
    assert lb <= val <= 1, f'{val} must be between [{lb}, 1]'
    if signed:
        yield (1 if val < 0 else 0)
        width -= 1
        val = abs(val)
    for _ in range(width):
        val = val * 2
        out_bit = np.floor(val)
        val = val - out_bit
        yield int(out_bit)