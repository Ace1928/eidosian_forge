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
def test_datetime2posixct_withNA(self):
    datetime = pandas.Series(pandas.date_range('2017-01-01 00:00:00.234', periods=20, freq='ms', tz='UTC'))
    datetime[1] = pandas.NaT
    with localconverter(default_converter + rpyp.converter) as cv:
        rp_c = robjects.conversion.converter_ctx.get().py2rpy(datetime)
        assert isinstance(rp_c, robjects.vectors.POSIXct)
        assert int(rp_c[0]) == 1483228800
        assert math.isnan(rp_c[1])