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
def test_nested_fields_are_records(self, nfields):
    """ Test that nested structured types are treated as records too """
    dt = np.dtype([('a', np.uint8), ('b', np.uint8), ('c', np.uint8)][:nfields])
    dt_outer = np.dtype([('inner', dt)])
    data = np.zeros(3, dt_outer).view(np.recarray)
    assert isinstance(data, np.recarray)
    assert isinstance(data['inner'], np.recarray)
    data0 = data[0]
    assert isinstance(data0, np.record)
    assert isinstance(data0['inner'], np.record)