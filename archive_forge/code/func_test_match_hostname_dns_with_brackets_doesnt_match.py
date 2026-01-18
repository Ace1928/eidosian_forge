import datetime
import mock
import pytest
from urllib3.connection import RECENT_DATE, CertificateError, _match_hostname
def test_match_hostname_dns_with_brackets_doesnt_match(self):
    cert = {'subjectAltName': (('DNS', 'localhost'), ('IP Address', 'localhost'))}
    asserted_hostname = '[localhost]'
    with pytest.raises(CertificateError) as e:
        _match_hostname(cert, asserted_hostname)
    assert "hostname '[localhost]' doesn't match either of 'localhost', 'localhost'" in str(e.value)