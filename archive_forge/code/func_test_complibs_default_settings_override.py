import os
import numpy as np
import pytest
from pandas.compat import (
from pandas.errors import (
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.io import pytables
from pandas.io.pytables import Term
def test_complibs_default_settings_override(tmp_path, setup_path):
    df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
    tmpfile = tmp_path / setup_path
    store = HDFStore(tmpfile)
    store.append('dfc', df, complevel=9, complib='blosc')
    store.append('df', df)
    store.close()
    with tables.open_file(tmpfile, mode='r') as h5file:
        for node in h5file.walk_nodes(where='/df', classname='Leaf'):
            assert node.filters.complevel == 0
            assert node.filters.complib is None
        for node in h5file.walk_nodes(where='/dfc', classname='Leaf'):
            assert node.filters.complevel == 9
            assert node.filters.complib == 'blosc'