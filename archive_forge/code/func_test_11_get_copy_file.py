import tempfile
from uuid import uuid1
import os.path as op
import os
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import pytest
from pyxnat.core import interfaces
@skip_if_no_network
def test_11_get_copy_file():
    fpath = op.join(tempfile.gettempdir(), uuid1().hex)
    fpath = subj_1.resource('test').file('hello.txt').get_copy(fpath)
    assert op.exists(fpath)
    fd = open(fpath, 'rb')
    try:
        assert fd.read() == bytes('Hello XNAT!\n', encoding='utf8')
    except TypeError:
        pass
    fd.close()
    os.remove(fpath)