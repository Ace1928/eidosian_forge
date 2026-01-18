import tempfile
from uuid import uuid1
import os.path as op
import os
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import pytest
from pyxnat.core import interfaces
@skip_if_no_network
def test_10_get_dir_file():
    fh = subj_1.resource('test').file('dir/hello.txt')
    fpath = fh.get()
    assert op.exists(fpath)
    try:
        assert open(fpath, 'rb').read() == bytes('Hello again!\n', encoding='utf8')
    except TypeError:
        pass
    custom = op.join(tempfile.gettempdir(), uuid1().hex)
    fh.get(custom)
    assert op.exists(custom), 'fpath: %s custom: %s' % (fpath, custom)
    os.remove(custom)
    os.remove(fpath)