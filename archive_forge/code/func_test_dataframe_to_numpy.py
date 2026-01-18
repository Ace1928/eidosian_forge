import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
def test_dataframe_to_numpy(self):
    df = robjects.vectors.DataFrame({'a': 1, 'b': 2, 'c': robjects.vectors.FactorVector('e'), 'd': robjects.vectors.StrVector(['xyz'])})
    with conversion.localconverter(robjects.default_converter + rpyn.converter) as cv:
        rec = cv.rpy2py(df)
    assert numpy.recarray == type(rec)
    assert rec.a[0] == 1
    assert rec.b[0] == 2
    assert rec.c[0] == 'e'
    assert rec.d[0] == 'xyz'