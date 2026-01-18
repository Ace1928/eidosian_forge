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
@pytest.mark.parametrize('rcode,names,values', (('data.frame(  a = c("a", "b"),  b = c(1, 2))', ('a', 'b'), (['a', 'b'], [1, 2])), ('data.frame(  a = c("a", "b"),  b = I(list(list(x=1, y=3), list(x=2, y=4))))', ('a', ('b', 'x'), ('b', 'y')), (['a', 'b'], [1, 2], [3, 4]))))
def test__flatten_dataframe(self, rcode, names, values):
    rdataf = robjects.r(rcode)
    colnames_lst = []
    columns = tuple(rpyp._flatten_dataframe(rdataf, colnames_lst))
    assert tuple(colnames_lst) == names