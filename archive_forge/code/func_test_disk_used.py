from __future__ import with_statement
import array
import os
from joblib.disk import disk_used, memstr_to_bytes, mkdirp, rm_subdirs
from joblib.testing import parametrize, raises
def test_disk_used(tmpdir):
    cachedir = tmpdir.strpath
    a = array.array('i')
    sizeof_i = a.itemsize
    target_size = 1024
    n = int(target_size * 1024 / sizeof_i)
    a = array.array('i', n * (1,))
    with open(os.path.join(cachedir, 'test'), 'wb') as output:
        a.tofile(output)
    assert disk_used(cachedir) >= target_size
    assert disk_used(cachedir) < target_size + 12