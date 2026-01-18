from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_bad_02(self):
    bad_dshape = '{ Unique Key : int64}'
    with pytest.raises(error.DataShapeSyntaxError):
        dshape(bad_dshape)