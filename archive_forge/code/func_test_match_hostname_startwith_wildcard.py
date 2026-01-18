import datetime
import mock
import pytest
from urllib3.connection import RECENT_DATE, CertificateError, _match_hostname
def test_match_hostname_startwith_wildcard(self):
    cert = {'subjectAltName': [('DNS', '*')]}
    asserted_hostname = 'foo'
    _match_hostname(cert, asserted_hostname)