from test import notPyPy2
import mock
import pytest
from urllib3.exceptions import SNIMissingWarning
from urllib3.util import ssl_
@pytest.mark.parametrize('addr', ['www.python.org', b'www.python.org', 'v2.sg.media-imdb.com', b'v2.sg.media-imdb.com'])
def test_is_ipaddress_false(addr):
    assert not ssl_.is_ipaddress(addr)