import numpy as np
def get_ufunc_info(ufunc_key):
    """get the lowering information for the ufunc with key ufunc_key.

    The lowering information is a dictionary that maps from a numpy
    loop string (as given by the ufunc types attribute) to a function
    that handles code generation for a scalar version of the ufunc
    (that is, generates the "per element" operation").

    raises a KeyError if the ufunc is not in the ufunc_db
    """
    _lazy_init_db()
    return _ufunc_db[ufunc_key]