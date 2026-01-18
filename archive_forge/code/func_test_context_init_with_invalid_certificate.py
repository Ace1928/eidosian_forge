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
def test_context_init_with_invalid_certificate(self, mock_utcnow, mock_log):
    mock_utcnow.return_value = datetime.datetime(2017, 1, 1)
    alt_cert_tuples = [('path', None)]
    context = certificate_utils.CertificateVerificationContext(alt_cert_tuples)
    self.assertEqual(0, len(context._signing_certificates))
    self.assertEqual(1, mock_log.error.call_count)