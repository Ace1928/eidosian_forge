from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography import x509, exceptions as cryptography_exceptions
from oslo_log import log as logging
from oslo_utils import timeutils
from cursive import exception
from cursive import signature_utils
from cursive import verifiers
def is_within_valid_dates(certificate):
    """Determine if the certificate is outside its valid date range.

    :param certificate: the cryptography certificate object
    :return: False if the certificate valid time range does not include
             now, True otherwise.
    """
    now = timeutils.utcnow()
    if now < certificate.not_valid_before:
        return False
    elif now > certificate.not_valid_after:
        return False
    return True