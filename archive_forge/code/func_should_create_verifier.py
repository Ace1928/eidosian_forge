import binascii
from castellan.common.exception import KeyManagerError
from castellan.common.exception import ManagedObjectNotFoundError
from castellan import key_manager
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import dsa
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography import x509
from oslo_log import log as logging
from oslo_serialization import base64
from oslo_utils import encodeutils
from cursive import exception
from cursive.i18n import _, _LE
from cursive import verifiers
def should_create_verifier(image_properties):
    """Determine whether a verifier should be created.

    Using the image properties, determine whether existing properties indicate
    that signature verification should be done.

    :param image_properties: the key-value properties about the image
    :return: True, if signature metadata properties exist, False otherwise
    """
    return image_properties is not None and CERT_UUID in image_properties and (HASH_METHOD in image_properties) and (SIGNATURE in image_properties) and (KEY_TYPE in image_properties)