import datetime
import mock
import pytest
from urllib3.connection import RECENT_DATE, CertificateError, _match_hostname
def test_match_hostname_empty_cert(self):
    cert = {}
    asserted_hostname = 'foo'
    with pytest.raises(ValueError):
        _match_hostname(cert, asserted_hostname)