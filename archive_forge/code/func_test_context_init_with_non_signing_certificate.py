import datetime
import mock
import os
from cryptography.hazmat.backends import default_backend
from cryptography import x509
from cursive import certificate_utils
from cursive import exception
from cursive.tests import base
@mock.patch('cursive.certificate_utils.LOG')
@mock.patch('oslo_utils.timeutils.utcnow')
def test_context_init_with_non_signing_certificate(self, mock_utcnow, mock_log):
    mock_utcnow.return_value = datetime.datetime(2017, 1, 1)
    non_signing_cert = self.load_certificate('self_signed_cert_missing_key_usage.pem')
    alt_cert_tuples = [('path', non_signing_cert)]
    context = certificate_utils.CertificateVerificationContext(alt_cert_tuples)
    self.assertEqual(0, len(context._signing_certificates))
    self.assertEqual(1, mock_log.warning.call_count)