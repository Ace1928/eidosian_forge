import inspect
import sys
import pytest
import numpy as np
from numpy.core import arange
from numpy.testing import assert_, assert_equal, assert_raises_regex
from numpy.lib import deprecate, deprecate_with_doc
import numpy.lib.utils as utils
from io import StringIO
@pytest.mark.parametrize('dtype', [np.dtype('i,i,i,i')[['f1', 'f3']], np.dtype('f8'), np.dtype('10i')])
def test_drop_metadata_identity_and_copy(dtype):
    assert utils.drop_metadata(dtype) is dtype
    dtype = np.dtype(dtype, metadata={1: 2})
    assert utils.drop_metadata(dtype).metadata is None