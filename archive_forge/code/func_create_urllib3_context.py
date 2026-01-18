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
def create_urllib3_context(ssl_version=None, cert_reqs=None, options=None, ciphers=None):
    """All arguments have the same meaning as ``ssl_wrap_socket``.

    By default, this function does a lot of the same work that
    ``ssl.create_default_context`` does on Python 3.4+. It:

    - Disables SSLv2, SSLv3, and compression
    - Sets a restricted set of server ciphers

    If you wish to enable SSLv3, you can do::

        from urllib3.util import ssl_
        context = ssl_.create_urllib3_context()
        context.options &= ~ssl_.OP_NO_SSLv3

    You can do the same to enable compression (substituting ``COMPRESSION``
    for ``SSLv3`` in the last line above).

    :param ssl_version:
        The desired protocol version to use. This will default to
        PROTOCOL_SSLv23 which will negotiate the highest protocol that both
        the server and your installation of OpenSSL support.
    :param cert_reqs:
        Whether to require the certificate verification. This defaults to
        ``ssl.CERT_REQUIRED``.
    :param options:
        Specific OpenSSL options. These default to ``ssl.OP_NO_SSLv2``,
        ``ssl.OP_NO_SSLv3``, ``ssl.OP_NO_COMPRESSION``, and ``ssl.OP_NO_TICKET``.
    :param ciphers:
        Which cipher suites to allow the server to select.
    :returns:
        Constructed SSLContext object with specified options
    :rtype: SSLContext
    """
    if not ssl_version or ssl_version == PROTOCOL_TLS:
        ssl_version = PROTOCOL_TLS_CLIENT
    context = SSLContext(ssl_version)
    context.set_ciphers(ciphers or DEFAULT_CIPHERS)
    cert_reqs = ssl.CERT_REQUIRED if cert_reqs is None else cert_reqs
    if options is None:
        options = 0
        options |= OP_NO_SSLv2
        options |= OP_NO_SSLv3
        options |= OP_NO_COMPRESSION
        options |= OP_NO_TICKET
    context.options |= options
    if (cert_reqs == ssl.CERT_REQUIRED or sys.version_info >= (3, 7, 4)) and getattr(context, 'post_handshake_auth', None) is not None:
        context.post_handshake_auth = True

    def disable_check_hostname():
        if getattr(context, 'check_hostname', None) is not None:
            context.check_hostname = False
    if cert_reqs == ssl.CERT_REQUIRED:
        context.verify_mode = cert_reqs
        disable_check_hostname()
    else:
        disable_check_hostname()
        context.verify_mode = cert_reqs
    if hasattr(context, 'keylog_filename'):
        sslkeylogfile = os.environ.get('SSLKEYLOGFILE')
        if sslkeylogfile:
            context.keylog_filename = sslkeylogfile
    return context