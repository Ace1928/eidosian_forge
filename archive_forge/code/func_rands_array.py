import string
import numpy as np
from pandas._typing import NpDtype
def rands_array(nchars, size, dtype: NpDtype='O', replace: bool=True) -> np.ndarray:
    """
    Generate an array of byte strings.
    """
    retval = np.random.choice(RANDS_CHARS, size=nchars * np.prod(size), replace=replace).view((np.str_, nchars)).reshape(size)
    return retval.astype(dtype)