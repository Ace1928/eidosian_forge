from __future__ import annotations
import warnings
from typing import Sequence
from cryptography.x509 import (
from cryptography.x509.extensions import ExtensionNotFound
from pyasn1.codec.der.decoder import decode
from pyasn1.type.char import IA5String
from .exceptions import CertificateError
from .hazmat import (
def extract_patterns(cert: Certificate) -> Sequence[CertificatePattern]:
    """
    Extract all valid ID patterns from a certificate for service verification.

    Args:
        cert: The certificate to be dissected.

    Returns:
        List of IDs.

    .. versionchanged:: 23.1.0
       ``commonName`` is not used as a fallback anymore.
    """
    ids: list[CertificatePattern] = []
    try:
        ext = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
    except ExtensionNotFound:
        pass
    else:
        ids.extend([DNSPattern.from_bytes(name.encode('utf-8')) for name in ext.value.get_values_for_type(DNSName)])
        ids.extend([URIPattern.from_bytes(uri.encode('utf-8')) for uri in ext.value.get_values_for_type(UniformResourceIdentifier)])
        ids.extend([IPAddressPattern(ip) for ip in ext.value.get_values_for_type(IPAddress)])
        for other in ext.value.get_values_for_type(OtherName):
            if other.type_id == ID_ON_DNS_SRV:
                srv, _ = decode(other.value)
                if isinstance(srv, IA5String):
                    ids.append(SRVPattern.from_bytes(srv.asOctets()))
                else:
                    raise CertificateError('Unexpected certificate content.')
    return ids