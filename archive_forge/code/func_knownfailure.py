from numpy.testing import dec
from nibabel.data import DataError
def knownfailure(f):
    return dec.knownfailureif(True)(f)