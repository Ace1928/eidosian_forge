import sys
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, IS_MUSL
from numpy.core.tests._locales import CommaDecimalPointLocale
from io import StringIO
@pytest.mark.parametrize('tp', [np.complex64, np.cdouble, np.clongdouble])
def test_complex_type_print(tp):
    """Check formatting when using print """
    for x in [0, 1, -1, 1e+20]:
        _test_redirected_print(complex(x), tp)
    if tp(1e+16).itemsize > 8:
        _test_redirected_print(complex(1e+16), tp)
    else:
        ref = '(1e+16+0j)'
        _test_redirected_print(complex(1e+16), tp, ref)
    _test_redirected_print(complex(np.inf, 1), tp, '(inf+1j)')
    _test_redirected_print(complex(-np.inf, 1), tp, '(-inf+1j)')
    _test_redirected_print(complex(-np.nan, 1), tp, '(nan+1j)')