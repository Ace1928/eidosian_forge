import datetime
import mock
import pytest
from urllib3.connection import RECENT_DATE, CertificateError, _match_hostname
def test_match_hostname_ip_address_ipv6(self):
    cert = {'subjectAltName': (('IP Address', '1:2::2:1'),)}
    asserted_hostname = '1:2::2:2'
    try:
        with mock.patch('urllib3.connection.log.warning') as mock_log:
            _match_hostname(cert, asserted_hostname)
    except CertificateError as e:
        assert "hostname '1:2::2:2' doesn't match '1:2::2:1'" in str(e)
        mock_log.assert_called_once_with('Certificate did not match expected hostname: %s. Certificate: %s', '1:2::2:2', {'subjectAltName': (('IP Address', '1:2::2:1'),)})
        assert e._peer_cert == cert