import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
@pytest.mark.parametrize('nfields', [0, 1, 2])
def test_assign_dtype_attribute(self, nfields):
    dt = np.dtype([('a', np.uint8), ('b', np.uint8), ('c', np.uint8)][:nfields])
    data = np.zeros(3, dt).view(np.recarray)
    assert data.dtype.type == np.record
    assert dt.type != np.record
    data.dtype = dt
    assert data.dtype.type == np.record