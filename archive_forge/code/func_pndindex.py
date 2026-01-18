import numpy as np
from numba.core.typing.typeof import typeof
from numba.core.typing.asnumbatype import as_numba_type
def pndindex(*args):
    """ Provides an n-dimensional parallel iterator that generates index tuples
    for each iteration point. Sequentially, pndindex is identical to np.ndindex.
    """
    return np.ndindex(*args)