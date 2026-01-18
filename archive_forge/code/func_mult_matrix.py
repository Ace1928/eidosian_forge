import io
import pathlib
import string
import struct
from html import escape
from typing import (
import charset_normalizer  # For str encoding detection
def mult_matrix(m1: Matrix, m0: Matrix) -> Matrix:
    a1, b1, c1, d1, e1, f1 = m1
    a0, b0, c0, d0, e0, f0 = m0
    'Returns the multiplication of two matrices.'
    return (a0 * a1 + c0 * b1, b0 * a1 + d0 * b1, a0 * c1 + c0 * d1, b0 * c1 + d0 * d1, a0 * e1 + c0 * f1 + e0, b0 * e1 + d0 * f1 + f0)