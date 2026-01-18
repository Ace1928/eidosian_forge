import datetime
import mock
import os
from cryptography.hazmat.backends import default_backend
from cryptography import x509
from cursive import certificate_utils
from cursive import exception
from cursive.tests import base
@mock.patch('oslo_utils.timeutils.utcnow')
def test_context_update_with_date_invalid_certificate(self, mock_utcnow):
    mock_utcnow.return_value = datetime.datetime(2017, 1, 1)
    certs = self.load_certificates(['self_signed_cert.pem', 'self_signed_cert.der'])
    cert_tuples = [('1', certs[0]), ('2', certs[1])]
    context = certificate_utils.CertificateVerificationContext(cert_tuples)
    cert = self.load_certificate('orphaned_cert.pem')
    mock_utcnow.return_value = datetime.datetime(2100, 1, 1)
    self.assertRaisesRegex(exception.SignatureVerificationError, 'The certificate is outside its valid date range.', context.update, cert)