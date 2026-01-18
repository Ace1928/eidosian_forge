import datetime
import mock
import os
from cryptography.hazmat.backends import default_backend
from cryptography import x509
from cursive import certificate_utils
from cursive import exception
from cursive.tests import base
def test_cannot_sign_certificates_without_key_usage(self):
    cert = self.load_certificate('self_signed_cert_missing_key_usage.pem')
    result = certificate_utils.can_sign_certificates(cert, 'test-ID')
    self.assertEqual(False, result)