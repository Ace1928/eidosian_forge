from typing import Union, Sequence
import pytest
import numpy as np
import cirq
def shift_matrix(width: int, shift: int) -> np.ndarray:
    result = np.zeros((width, width))
    for i in range(width):
        result[(i + shift) % width, i] = 1
    return result