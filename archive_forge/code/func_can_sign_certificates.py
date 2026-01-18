from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography import x509, exceptions as cryptography_exceptions
from oslo_log import log as logging
from oslo_utils import timeutils
from cursive import exception
from cursive import signature_utils
from cursive import verifiers
def can_sign_certificates(certificate, certificate_uuid=''):
    """Determine if the certificate can sign other certificates.

    :param certificate: the cryptography certificate object
    :param certificate_uuid: the uuid of the certificate
    :return: False if the certificate cannot sign other certificates,
             True otherwise.
    """
    try:
        basic_constraints = certificate.extensions.get_extension_for_oid(x509.oid.ExtensionOID.BASIC_CONSTRAINTS).value
    except x509.extensions.ExtensionNotFound:
        LOG.debug("Certificate '%s' does not have a basic constraints extension.", certificate_uuid)
        return False
    try:
        key_usage = certificate.extensions.get_extension_for_oid(x509.oid.ExtensionOID.KEY_USAGE).value
    except x509.extensions.ExtensionNotFound:
        LOG.debug("Certificate '%s' does not have a key usage extension.", certificate_uuid)
        return False
    if basic_constraints.ca and key_usage.key_cert_sign:
        return True
    if not basic_constraints.ca:
        LOG.debug("Certificate '%s' is not marked as a CA in its basic constraints extension.", certificate_uuid)
    if not key_usage.key_cert_sign:
        LOG.debug("Certificate '%s' is not marked for verifying certificate signatures in its key usage extension.", certificate_uuid)
    return False