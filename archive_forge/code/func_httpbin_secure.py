import ssl
import tempfile
import threading
import pytest
from requests.compat import urljoin
@pytest.fixture
def httpbin_secure(httpbin_secure):
    return prepare_url(httpbin_secure)