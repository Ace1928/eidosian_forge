import sys
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, IS_MUSL
from numpy.core.tests._locales import CommaDecimalPointLocale
from io import StringIO
@pytest.mark.parametrize('tp', [np.float32, np.double, np.longdouble])
def test_float_types(tp):
    """ Check formatting.

        This is only for the str function, and only for simple types.
        The precision of np.float32 and np.longdouble aren't the same as the
        python float precision.

    """
    for x in [0, 1, -1, 1e+20]:
        assert_equal(str(tp(x)), str(float(x)), err_msg='Failed str formatting for type %s' % tp)
    if tp(1e+16).itemsize > 4:
        assert_equal(str(tp(1e+16)), str(float('1e16')), err_msg='Failed str formatting for type %s' % tp)
    else:
        ref = '1e+16'
        assert_equal(str(tp(1e+16)), ref, err_msg='Failed str formatting for type %s' % tp)