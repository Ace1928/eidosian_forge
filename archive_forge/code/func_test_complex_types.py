import sys
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, IS_MUSL
from numpy.core.tests._locales import CommaDecimalPointLocale
from io import StringIO
@pytest.mark.parametrize('tp', [np.complex64, np.cdouble, np.clongdouble])
def test_complex_types(tp):
    """Check formatting of complex types.

        This is only for the str function, and only for simple types.
        The precision of np.float32 and np.longdouble aren't the same as the
        python float precision.

    """
    for x in [0, 1, -1, 1e+20]:
        assert_equal(str(tp(x)), str(complex(x)), err_msg='Failed str formatting for type %s' % tp)
        assert_equal(str(tp(x * 1j)), str(complex(x * 1j)), err_msg='Failed str formatting for type %s' % tp)
        assert_equal(str(tp(x + x * 1j)), str(complex(x + x * 1j)), err_msg='Failed str formatting for type %s' % tp)
    if tp(1e+16).itemsize > 8:
        assert_equal(str(tp(1e+16)), str(complex(1e+16)), err_msg='Failed str formatting for type %s' % tp)
    else:
        ref = '(1e+16+0j)'
        assert_equal(str(tp(1e+16)), ref, err_msg='Failed str formatting for type %s' % tp)