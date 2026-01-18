from typing import (
import cmath
import re
import numpy as np
import sympy
def parse_matrix(text: str) -> np.ndarray:
    """Attempts to parse a complex matrix in exactly the same way as Quirk."""
    text = re.sub('\\s', '', text)
    if len(text) < 4 or text[:2] != '{{' or text[-2:] != '}}':
        raise ValueError('Not surrounded by {{}}.')
    return np.array([[parse_complex(cell) for cell in row.split(',')] for row in text[2:-2].split('},{')], dtype=np.complex128)