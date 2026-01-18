import base64
import datetime
import mock
from castellan.common.exception import KeyManagerError
from castellan.common.exception import ManagedObjectNotFoundError
import cryptography.exceptions as crypto_exceptions
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import dsa
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from oslo_utils import timeutils
from cursive import exception
from cursive import signature_utils
from cursive.tests import base
@mock.patch('castellan.key_manager.API', return_value=FakeKeyManager())
def test_get_certificate_id_not_exist(self, mock_key_manager):
    bad_cert_uuid = 'invalid-cert-uuid'
    self.assertRaisesRegex(exception.SignatureVerificationError, 'Certificate not found with ID: .*', signature_utils.get_certificate, None, bad_cert_uuid)