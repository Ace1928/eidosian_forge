import pytest
import textwrap
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from numpy.f2py.tests import util
def test_array_input_varia(self):
    f = getattr(self.module, self.fprefix + '_array_input')
    assert_array_equal(f(['a', 'b', 'c']), np.array(list(map(ord, 'abc')), dtype='i1'))
    assert_array_equal(f([b'a', b'b', b'c']), np.array(list(map(ord, 'abc')), dtype='i1'))
    try:
        f(['a', 'b', 'c', 'd'])
    except ValueError as msg:
        if not str(msg).endswith('th dimension must be fixed to 3 but got 4'):
            raise
    else:
        raise SystemError(f'{f.__name__} should have failed on wrong input')