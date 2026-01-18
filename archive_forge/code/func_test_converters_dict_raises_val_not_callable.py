import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_converters_dict_raises_val_not_callable():
    with pytest.raises(TypeError, match='values of the converters dictionary must be callable'):
        np.loadtxt(StringIO('1 2\n3 4'), converters={0: 1})