import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
@pytest.mark.skipif(string_to_longdouble_inaccurate, reason='Need strtold_l')
@pytest.mark.parametrize('strtype', (np.str_, np.bytes_, str, bytes))
def test_array_and_stringlike_roundtrip(strtype):
    """
    Test that string representations of long-double roundtrip both
    for array casting and scalar coercion, see also gh-15608.
    """
    o = 1 + LD_INFO.eps
    if strtype in (np.bytes_, bytes):
        o_str = strtype(repr(o).encode('ascii'))
    else:
        o_str = strtype(repr(o))
    assert o == np.longdouble(o_str)
    o_strarr = np.asarray([o] * 3, dtype=strtype)
    assert (o == o_strarr.astype(np.longdouble)).all()
    assert (o_strarr == o_str).all()
    assert (np.asarray([o] * 3).astype(strtype) == o_str).all()