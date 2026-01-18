from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography import x509, exceptions as cryptography_exceptions
from oslo_log import log as logging
from oslo_utils import timeutils
from cursive import exception
from cursive import signature_utils
from cursive import verifiers
def is_issuer(issuing_certificate, issued_certificate):
    """Determine if the issuing cert is the parent of the issued cert.

    Determine if the issuing certificate is the parent of the issued
    certificate by:
    * conducting subject and issuer name matching, and
    * verifying the signature of the issued certificate with the issuing
      certificate's public key

    :param issuing_certificate: the cryptography certificate object that
           is the potential parent of the issued certificate
    :param issued_certificate: the cryptography certificate object that
           is the potential child of the issuing certificate
    :return: True if the issuing certificate is the parent of the issued
             certificate, False otherwise.
    """
    if issuing_certificate is None or issued_certificate is None:
        return False
    elif issuing_certificate.subject != issued_certificate.issuer:
        return False
    else:
        try:
            verify_certificate_signature(issuing_certificate, issued_certificate)
        except cryptography_exceptions.InvalidSignature:
            return False
        return True