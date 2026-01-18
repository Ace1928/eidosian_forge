import dataclasses
import datetime
import platform
import ssl
import typing
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
def default_tls_context(usage: str='initiate') -> CredSSPTLSContext:
    """CredSSP TLSContext with sane defaults.

    Creates the TLS context used to generate the SSL object for CredSSP
    authentication. By default the TLS context will set the minimum protocol to
    TLSv1.2. Certificate verification is also disabled for both the initiator
    and acceptor as per the `MS-CSSP Events and Sequencing Rules`_ in step 1.
    This can be used as a base context where the caller applies further changes
    based on their requirements such as cert validation and so forth.

    This context is then passed in through the `credssp_tls_context` kwarg of
    :meth:`spnego.client` or :meth:`spnego.server`.

    Args:
        usage: Either `initiate` for a client context or `accept` for a server context.

    Returns:
        TLSContext: The TLS context that can be used with CredSSP auth.

    .. _MS-CSSP Events and Sequencing Rules:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-cssp/385a7489-d46b-464c-b224-f7340e308a5c
    """
    if usage == 'initiate':
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    else:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.options |= ssl.OP_NO_COMPRESSION | 512 | 2048
    tls_version = getattr(ssl, 'TLSVersion', None)
    if hasattr(ctx, 'minimum_version') and tls_version:
        setattr(ctx, 'minimum_version', tls_version.TLSv1_2)
    else:
        ctx.options |= ssl.Options.OP_NO_SSLv2 | ssl.Options.OP_NO_SSLv3 | ssl.Options.OP_NO_TLSv1 | ssl.Options.OP_NO_TLSv1_1
    return CredSSPTLSContext(context=ctx)