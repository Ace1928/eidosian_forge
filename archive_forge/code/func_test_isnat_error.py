import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_isnat_error(self):
    for t in np.typecodes['All']:
        if t in np.typecodes['Datetime']:
            continue
        assert_raises(TypeError, np.isnat, np.zeros(10, t))