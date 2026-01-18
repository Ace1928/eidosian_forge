import string
import numpy as np
from pandas._typing import NpDtype
def rands(nchars) -> str:
    """
    Generate one random byte string.

    See `rands_array` if you want to create an array of random strings.

    """
    return ''.join(np.random.choice(RANDS_CHARS, nchars))