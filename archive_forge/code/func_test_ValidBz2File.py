import os
import pytest
from tempfile import mkdtemp, mkstemp, NamedTemporaryFile
from shutil import rmtree
import numpy.lib._datasource as datasource
from numpy.testing import assert_, assert_equal, assert_raises
import urllib.request as urllib_request
from urllib.parse import urlparse
from urllib.error import URLError
def test_ValidBz2File(self):
    try:
        import bz2
    except ImportError:
        pytest.skip()
    filepath = os.path.join(self.tmpdir, 'foobar.txt.bz2')
    fp = bz2.BZ2File(filepath, 'w')
    fp.write(magic_line)
    fp.close()
    fp = self.ds.open(filepath)
    result = fp.readline()
    fp.close()
    assert_equal(magic_line, result)