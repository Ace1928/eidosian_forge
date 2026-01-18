import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_comparisons_return_not_implemented():

    class custom:
        __array_priority__ = 10000
    obj = custom()
    dt = np.datetime64('2000', 'ns')
    td = dt - dt
    for item in [dt, td]:
        assert item.__eq__(obj) is NotImplemented
        assert item.__ne__(obj) is NotImplemented
        assert item.__le__(obj) is NotImplemented
        assert item.__lt__(obj) is NotImplemented
        assert item.__ge__(obj) is NotImplemented
        assert item.__gt__(obj) is NotImplemented