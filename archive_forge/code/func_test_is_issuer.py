import datetime
import mock
import os
from cryptography.hazmat.backends import default_backend
from cryptography import x509
from cursive import certificate_utils
from cursive import exception
from cursive.tests import base
def test_is_issuer(self):
    cert = self.load_certificate('self_signed_cert.pem')
    result = certificate_utils.is_issuer(cert, cert)
    self.assertEqual(True, result)