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
def get_verifier(context, img_signature_certificate_uuid, img_signature_hash_method, img_signature, img_signature_key_type):
    """Instantiate signature properties and use them to create a verifier.

    :param context: the user context for authentication
    :param img_signature_certificate_uuid:
           uuid of signing certificate stored in key manager
    :param img_signature_hash_method:
           string denoting hash method used to compute signature
    :param img_signature: string of base64 encoding of signature
    :param img_signature_key_type:
           string denoting type of keypair used to compute signature
    :returns: instance of
       cryptography.hazmat.primitives.asymmetric.AsymmetricVerificationContext
    :raises: SignatureVerificationError if we fail to build the verifier
    """
    image_meta_props = {'img_signature_uuid': img_signature_certificate_uuid, 'img_signature_hash_method': img_signature_hash_method, 'img_signature': img_signature, 'img_signature_key_type': img_signature_key_type}
    for key in image_meta_props.keys():
        if image_meta_props[key] is None:
            raise exception.SignatureVerificationError(reason=_('Required image properties for signature verification do not exist. Cannot verify signature. Missing property: %s') % key)
    signature = get_signature(img_signature)
    hash_method = get_hash_method(img_signature_hash_method)
    signature_key_type = SignatureKeyType.lookup(img_signature_key_type)
    public_key = get_public_key(context, img_signature_certificate_uuid, signature_key_type)
    verifier = signature_key_type.create_verifier(signature, hash_method, public_key)
    if verifier:
        return verifier
    else:
        raise exception.SignatureVerificationError(reason=_('Error occurred while creating the verifier'))