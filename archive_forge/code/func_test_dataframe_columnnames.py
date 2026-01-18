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
def test_dataframe_columnnames(self):
    pd_df = pandas.DataFrame({'the one': [1, 2], 'the other': [3, 4]})
    with localconverter(default_converter + rpyp.converter) as cv:
        rp_df = robjects.conversion.converter_ctx.get().py2rpy(pd_df)
    assert tuple(rp_df.names) == ('the one', 'the other')