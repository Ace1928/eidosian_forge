from __future__ import absolute_import
import hmac
import os
import sys
import warnings
from binascii import hexlify, unhexlify
from hashlib import md5, sha1, sha256
from ..exceptions import (
from ..packages import six
from .url import BRACELESS_IPV6_ADDRZ_RE, IPV4_RE
def ssl_wrap_socket(sock, keyfile=None, certfile=None, cert_reqs=None, ca_certs=None, server_hostname=None, ssl_version=None, ciphers=None, ssl_context=None, ca_cert_dir=None, key_password=None, ca_cert_data=None, tls_in_tls=False):
    """
    All arguments except for server_hostname, ssl_context, and ca_cert_dir have
    the same meaning as they do when using :func:`ssl.wrap_socket`.

    :param server_hostname:
        When SNI is supported, the expected hostname of the certificate
    :param ssl_context:
        A pre-made :class:`SSLContext` object. If none is provided, one will
        be created using :func:`create_urllib3_context`.
    :param ciphers:
        A string of ciphers we wish the client to support.
    :param ca_cert_dir:
        A directory containing CA certificates in multiple separate files, as
        supported by OpenSSL's -CApath flag or the capath argument to
        SSLContext.load_verify_locations().
    :param key_password:
        Optional password if the keyfile is encrypted.
    :param ca_cert_data:
        Optional string containing CA certificates in PEM format suitable for
        passing as the cadata parameter to SSLContext.load_verify_locations()
    :param tls_in_tls:
        Use SSLTransport to wrap the existing socket.
    """
    context = ssl_context
    if context is None:
        context = create_urllib3_context(ssl_version, cert_reqs, ciphers=ciphers)
    if ca_certs or ca_cert_dir or ca_cert_data:
        try:
            context.load_verify_locations(ca_certs, ca_cert_dir, ca_cert_data)
        except (IOError, OSError) as e:
            raise SSLError(e)
    elif ssl_context is None and hasattr(context, 'load_default_certs'):
        context.load_default_certs()
    if keyfile and key_password is None and _is_key_file_encrypted(keyfile):
        raise SSLError('Client private key is encrypted, password is required')
    if certfile:
        if key_password is None:
            context.load_cert_chain(certfile, keyfile)
        else:
            context.load_cert_chain(certfile, keyfile, key_password)
    try:
        if hasattr(context, 'set_alpn_protocols'):
            context.set_alpn_protocols(ALPN_PROTOCOLS)
    except NotImplementedError:
        pass
    use_sni_hostname = server_hostname and (not is_ipaddress(server_hostname))
    send_sni = use_sni_hostname and HAS_SNI or (IS_SECURETRANSPORT and server_hostname)
    if not HAS_SNI and use_sni_hostname:
        warnings.warn('An HTTPS request has been made, but the SNI (Server Name Indication) extension to TLS is not available on this platform. This may cause the server to present an incorrect TLS certificate, which can cause validation failures. You can upgrade to a newer version of Python to solve this. For more information, see https://urllib3.readthedocs.io/en/1.26.x/advanced-usage.html#ssl-warnings', SNIMissingWarning)
    if send_sni:
        ssl_sock = _ssl_wrap_socket_impl(sock, context, tls_in_tls, server_hostname=server_hostname)
    else:
        ssl_sock = _ssl_wrap_socket_impl(sock, context, tls_in_tls)
    return ssl_sock