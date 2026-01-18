import os
import pytest
from tempfile import mkdtemp, mkstemp, NamedTemporaryFile
from shutil import rmtree
import numpy.lib._datasource as datasource
from numpy.testing import assert_, assert_equal, assert_raises
import urllib.request as urllib_request
from urllib.parse import urlparse
from urllib.error import URLError
def test_CachedHTTPFile(self):
    localfile = valid_httpurl()
    scheme, netloc, upath, pms, qry, frg = urlparse(localfile)
    local_path = os.path.join(self.repos._destpath, netloc)
    os.mkdir(local_path, 448)
    tmpfile = valid_textfile(local_path)
    assert_(self.repos.exists(tmpfile))