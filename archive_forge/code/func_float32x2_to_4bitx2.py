from typing import Tuple, Union
import numpy as np
def float32x2_to_4bitx2(val_low: np.dtype, val_high: np.dtype, signed: bool) -> np.ndarray:
    """Cast two elements to 4bit (via rounding and clipping) and pack
    to a single byte
    Args:
        val_low: element to be packed in the 4 LSB
        val_high: element to be packed in the 4 MSB
        signed: boolean, whether to convert to signed int4.

    Returns:
        An ndarray with a single int8/uint8 element, containing both int4 elements
    """
    i8_high = float32_to_4bit_unpacked(val_high, signed)
    i8_low = float32_to_4bit_unpacked(val_low, signed)
    return i8_high << 4 | i8_low & 15