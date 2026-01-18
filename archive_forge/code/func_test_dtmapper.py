import numpy as np
import pytest
from ..volumeutils import DtypeMapper, Recoder, native_code, swapped_code
def test_dtmapper():
    d = DtypeMapper()
    with pytest.raises(KeyError):
        d[1]
    d[1] = 'something'
    assert d[1] == 'something'
    assert list(d.keys()) == [1]
    assert list(d.values()) == ['something']
    intp_dt = np.dtype('intp')
    if intp_dt == np.dtype('int32'):
        canonical_dt = np.dtype('int32')
    elif intp_dt == np.dtype('int64'):
        canonical_dt = np.dtype('int64')
    else:
        raise RuntimeError('Can I borrow your computer?')
    native_dt = canonical_dt.newbyteorder('=')
    explicit_dt = canonical_dt.newbyteorder(native_code)
    d[canonical_dt] = 'spam'
    assert d[canonical_dt] == 'spam'
    assert d[native_dt] == 'spam'
    assert d[explicit_dt] == 'spam'
    d = DtypeMapper()
    assert list(d.keys()) == []
    assert list(d.keys()) == []
    d[canonical_dt] = 'spam'
    assert list(d.keys()) == [canonical_dt]
    assert list(d.values()) == ['spam']
    d = DtypeMapper()
    sw_dt = canonical_dt.newbyteorder(swapped_code)
    d[sw_dt] = 'spam'
    with pytest.raises(KeyError):
        d[canonical_dt]
    assert d[sw_dt] == 'spam'
    sw_intp_dt = intp_dt.newbyteorder(swapped_code)
    assert d[sw_intp_dt] == 'spam'