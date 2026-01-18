import tempfile
from uuid import uuid1
import os.path as op
import os
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import pytest
from pyxnat.core import interfaces
@skip_if_no_network
def test_07_put_file():
    local_path = op.join(_modulepath, 'hello_xnat.txt')
    f = subj_1.resource('test').file('hello.txt')
    subj_1.resource('test').file('hello.txt').put(local_path)
    subj_1.resource('test').put([local_path])
    assert f.exists()
    assert int(f.size()) == os.stat(local_path).st_size