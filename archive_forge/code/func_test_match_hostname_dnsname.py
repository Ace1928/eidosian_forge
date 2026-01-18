import datetime
import mock
import pytest
from urllib3.connection import RECENT_DATE, CertificateError, _match_hostname
def test_match_hostname_dnsname(self):
    cert = {'subjectAltName': [('DNS', 'xn--p1b6ci4b4b3a*.xn--11b5bs8d')]}
    asserted_hostname = 'xn--p1b6ci4b4b3a*.xn--11b5bs8d'
    _match_hostname(cert, asserted_hostname)