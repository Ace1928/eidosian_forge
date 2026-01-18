import math
from collections import OrderedDict
from datetime import datetime
import pytest
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import vectors
from rpy2.robjects import conversion
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
@pytest.mark.parametrize('dtype', (pandas.Int32Dtype() if has_pandas else None, pandas.Int64Dtype() if has_pandas else None))
def test_dataframe_int_nan(self, dtype):
    a = pandas.DataFrame([(numpy.NaN,)], dtype=dtype, columns=['z'])
    with localconverter(default_converter + rpyp.converter) as cv:
        b = robjects.conversion.get_conversion().py2rpy(a)
    assert b[0][0] is rinterface.na_values.NA_Integer
    with localconverter(default_converter + rpyp.converter) as cv:
        c = robjects.conversion.get_conversion().rpy2py(b)