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
def test_get_hash_method(self):
    hash_dict = signature_utils.HASH_METHODS
    for hash_name in hash_dict.keys():
        hash_class = signature_utils.get_hash_method(hash_name).__class__
        self.assertIsInstance(hash_dict[hash_name], hash_class)