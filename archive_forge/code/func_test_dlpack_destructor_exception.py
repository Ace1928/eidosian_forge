import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, IS_PYPY
def test_dlpack_destructor_exception(self):
    with pytest.raises(RuntimeError):
        self.dlpack_deleter_exception()