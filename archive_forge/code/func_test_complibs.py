import os
import numpy as np
import pytest
from pandas.compat import (
from pandas.errors import (
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.io import pytables
from pandas.io.pytables import Term
@pytest.mark.parametrize('lvl', range(10))
@pytest.mark.parametrize('lib', tables.filters.all_complibs)
@pytest.mark.filterwarnings('ignore:object name is not a valid')
@pytest.mark.skipif(not PY311 and is_ci_environment() and is_platform_linux(), reason='Segfaulting in a CI environment')
def test_complibs(tmp_path, lvl, lib, request):
    if PY311 and is_platform_linux() and (lib == 'blosc2') and (lvl != 0):
        request.applymarker(pytest.mark.xfail(reason=f'Fails for {lib} on Linux and PY > 3.11'))
    df = DataFrame(np.ones((30, 4)), columns=list('ABCD'), index=np.arange(30).astype(np.str_))
    if not tables.which_lib_version('lzo'):
        pytest.skip('lzo not available')
    if not tables.which_lib_version('bzip2'):
        pytest.skip('bzip2 not available')
    tmpfile = tmp_path / f'{lvl}_{lib}.h5'
    gname = f'{lvl}_{lib}'
    df.to_hdf(tmpfile, key=gname, complib=lib, complevel=lvl)
    result = read_hdf(tmpfile, gname)
    tm.assert_frame_equal(result, df)
    with tables.open_file(tmpfile, mode='r') as h5table:
        for node in h5table.walk_nodes(where='/' + gname, classname='Leaf'):
            assert node.filters.complevel == lvl
            if lvl == 0:
                assert node.filters.complib is None
            else:
                assert node.filters.complib == lib