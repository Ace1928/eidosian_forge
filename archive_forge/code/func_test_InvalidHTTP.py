import os
import pytest
from tempfile import mkdtemp, mkstemp, NamedTemporaryFile
from shutil import rmtree
import numpy.lib._datasource as datasource
from numpy.testing import assert_, assert_equal, assert_raises
import urllib.request as urllib_request
from urllib.parse import urlparse
from urllib.error import URLError
def test_InvalidHTTP(self):
    scheme, netloc, upath, pms, qry, frg = urlparse(invalid_httpurl())
    invalidhttp = os.path.join(self.tmpdir, netloc, upath.strip(os.sep).strip('/'))
    assert_(invalidhttp != self.ds.abspath(valid_httpurl()))