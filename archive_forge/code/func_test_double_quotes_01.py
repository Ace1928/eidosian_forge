from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_double_quotes_01(self):
    quotes_dshape = '{ \'doublequote " field "\' : int64 }'
    ds1 = dshape(quotes_dshape)
    ds2 = dshape(str(ds1))
    assert str(ds1) == str(ds2)