import ctypes
import re
from typing import Any, Optional, Tuple, Union
import numpy as np
import pandas
from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
from modin.core.dataframe.base.interchange.dataframe_protocol.utils import (
def bitmask_to_bool_ndarray(bitmask: np.ndarray, mask_length: int, first_byte_offset: int=0) -> np.ndarray:
    """
    Convert bit-mask to a boolean NumPy array.

    Parameters
    ----------
    bitmask : np.ndarray[uint8]
        NumPy array of uint8 dtype representing the bitmask.
    mask_length : int
        Number of elements in the mask to interpret.
    first_byte_offset : int, default: 0
        Number of elements to offset from the start of the first byte.

    Returns
    -------
    np.ndarray[bool]
    """
    bytes_to_skip = first_byte_offset // 8
    bitmask = bitmask[bytes_to_skip:]
    first_byte_offset %= 8
    bool_mask = np.zeros(mask_length, dtype=bool)
    val = bitmask[0]
    mask_idx = 0
    bits_in_first_byte = min(8 - first_byte_offset, mask_length)
    for j in range(bits_in_first_byte):
        if val & 1 << j + first_byte_offset:
            bool_mask[mask_idx] = True
        mask_idx += 1
    for i in range((mask_length - bits_in_first_byte) // 8):
        val = bitmask[i + 1]
        for j in range(8):
            if val & 1 << j:
                bool_mask[mask_idx] = True
            mask_idx += 1
    if len(bitmask) > 1:
        val = bitmask[-1]
        for j in range(len(bool_mask) - mask_idx):
            if val & 1 << j:
                bool_mask[mask_idx] = True
            mask_idx += 1
    return bool_mask