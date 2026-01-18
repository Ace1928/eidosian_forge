from numpy.testing import dec
from nibabel.data import DataError
def skip_func(func):
    return dec.skipif(True, msg)(func)