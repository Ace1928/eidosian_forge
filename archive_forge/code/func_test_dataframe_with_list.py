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
@pytest.mark.parametrize('rcode,names,values', (('data.frame(  a = c("a", "b"),  b = I(list(list(x=1, y=3), list(x=2, y=4))))', ('a', ('b', 'x'), ('b', 'y')), (['a', 'b'], [1, 2], [3, 4])), ('data.frame(  b = I(list(list(x=1, y=3), list(x=2, y=4))))', (('b', 'x'), ('b', 'y')), ([1, 2], [3, 4])), ('aggregate(gear ~ am, mtcars, function(x) c(mean = mean(x), sd = sd(x)), simplify=FALSE)', ('am', ('gear', 'mean'), ('gear', 'sd')), ([0.0, 1.0], [3.210526, 4.384615], [0.418854, 0.50637]))))
def test_dataframe_with_list(self, rcode, names, values):
    rdataf = robjects.r(rcode)
    with localconverter(default_converter + rpyp.converter) as cv:
        pandas_df = robjects.conversion.converter_ctx.get().rpy2py(rdataf)
    assert tuple(pandas_df.columns) == names