from numpy.testing import dec
from nibabel.data import DataError
def if_datasource(ds, msg):
    try:
        ds.get_filename()
    except DataError:
        return dec.skipif(True, msg)
    return lambda f: f