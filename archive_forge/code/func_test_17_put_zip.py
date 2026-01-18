import tempfile
from uuid import uuid1
import os.path as op
import os
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import pytest
from pyxnat.core import interfaces
@skip_if_no_network
def test_17_put_zip():
    local_path = op.join(_modulepath, 'hello_dir.zip')
    assert op.exists(local_path)
    r1 = subj_1.resource('test_zip_extract')
    r1.put_zip(local_path, extract=True)
    assert r1.exists()
    assert r1.file('hello_dir/hello_xnat_dir.txt').exists()
    assert r1.file('hello_dir/hello_dir2/hello_xnat_dir2.txt').exists()
    r2 = subj_1.resource('test_zip_no_extract')
    r2.put_zip(local_path, extract=False)
    assert r2.exists()
    assert r2.file('hello_dir.zip').exists()