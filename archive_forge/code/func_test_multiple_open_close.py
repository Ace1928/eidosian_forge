import os
import numpy as np
import pytest
from pandas.compat import (
from pandas.errors import (
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.io import pytables
from pandas.io.pytables import Term
def test_multiple_open_close(tmp_path, setup_path):
    path = tmp_path / setup_path
    df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
    df.to_hdf(path, key='df', mode='w', format='table')
    store = HDFStore(path)
    assert 'CLOSED' not in store.info()
    assert store.is_open
    store.close()
    assert 'CLOSED' in store.info()
    assert not store.is_open
    path = tmp_path / setup_path
    if pytables._table_file_open_policy_is_strict:
        store1 = HDFStore(path)
        msg = 'The file [\\S]* is already opened\\.  Please close it before reopening in write mode\\.'
        with pytest.raises(ValueError, match=msg):
            HDFStore(path)
        store1.close()
    else:
        store1 = HDFStore(path)
        store2 = HDFStore(path)
        assert 'CLOSED' not in store1.info()
        assert 'CLOSED' not in store2.info()
        assert store1.is_open
        assert store2.is_open
        store1.close()
        assert 'CLOSED' in store1.info()
        assert not store1.is_open
        assert 'CLOSED' not in store2.info()
        assert store2.is_open
        store2.close()
        assert 'CLOSED' in store1.info()
        assert 'CLOSED' in store2.info()
        assert not store1.is_open
        assert not store2.is_open
        store = HDFStore(path, mode='w')
        store.append('df', df)
        store2 = HDFStore(path)
        store2.append('df2', df)
        store2.close()
        assert 'CLOSED' in store2.info()
        assert not store2.is_open
        store.close()
        assert 'CLOSED' in store.info()
        assert not store.is_open
        store = HDFStore(path, mode='w')
        store.append('df', df)
        store2 = HDFStore(path)
        store.close()
        assert 'CLOSED' in store.info()
        assert not store.is_open
        store2.close()
        assert 'CLOSED' in store2.info()
        assert not store2.is_open
    path = tmp_path / setup_path
    df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
    df.to_hdf(path, key='df', mode='w', format='table')
    store = HDFStore(path)
    store.close()
    msg = '[\\S]* file is not open!'
    with pytest.raises(ClosedFileError, match=msg):
        store.keys()
    with pytest.raises(ClosedFileError, match=msg):
        'df' in store
    with pytest.raises(ClosedFileError, match=msg):
        len(store)
    with pytest.raises(ClosedFileError, match=msg):
        store['df']
    with pytest.raises(ClosedFileError, match=msg):
        store.select('df')
    with pytest.raises(ClosedFileError, match=msg):
        store.get('df')
    with pytest.raises(ClosedFileError, match=msg):
        store.append('df2', df)
    with pytest.raises(ClosedFileError, match=msg):
        store.put('df3', df)
    with pytest.raises(ClosedFileError, match=msg):
        store.get_storer('df2')
    with pytest.raises(ClosedFileError, match=msg):
        store.remove('df2')
    with pytest.raises(ClosedFileError, match=msg):
        store.select('df')
    msg = "'HDFStore' object has no attribute 'df'"
    with pytest.raises(AttributeError, match=msg):
        store.df