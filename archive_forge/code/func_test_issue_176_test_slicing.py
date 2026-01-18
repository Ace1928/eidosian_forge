from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_176_test_slicing(self):
    from srsly.ruamel_yaml.compat import PY2
    mss = round_trip_load('[0, 1, 2, 3, 4]')
    assert len(mss) == 5
    assert mss[2:2] == []
    assert mss[2:4] == [2, 3]
    assert mss[1::2] == [1, 3]
    m = mss[:]
    m[2:2] = [42]
    assert m == [0, 1, 42, 2, 3, 4]
    m = mss[:]
    m[:3] = [42, 43, 44]
    assert m == [42, 43, 44, 3, 4]
    m = mss[:]
    m[2:] = [42, 43, 44]
    assert m == [0, 1, 42, 43, 44]
    m = mss[:]
    m[:] = [42, 43, 44]
    assert m == [42, 43, 44]
    m = mss[:]
    m[2:4] = [42, 43, 44]
    assert m == [0, 1, 42, 43, 44, 4]
    m = mss[:]
    m[1::2] = [42, 43]
    assert m == [0, 42, 2, 43, 4]
    m = mss[:]
    if PY2:
        with pytest.raises(ValueError, match='attempt to assign'):
            m[1::2] = [42, 43, 44]
    else:
        with pytest.raises(TypeError, match='too many'):
            m[1::2] = [42, 43, 44]
    if PY2:
        with pytest.raises(ValueError, match='attempt to assign'):
            m[1::2] = [42]
    else:
        with pytest.raises(TypeError, match='not enough'):
            m[1::2] = [42]
    m = mss[:]
    m += [5]
    m[1::2] = [42, 43, 44]
    assert m == [0, 42, 2, 43, 4, 44]
    m = mss[:]
    del m[1:3]
    assert m == [0, 3, 4]
    m = mss[:]
    del m[::2]
    assert m == [1, 3]
    m = mss[:]
    del m[:]
    assert m == []