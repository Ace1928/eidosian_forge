from __future__ import print_function
import logging
import os
import socket
import ssl
import sys
import threading
import warnings
from datetime import datetime
import tornado.httpserver
import tornado.ioloop
import tornado.netutil
import tornado.web
import trustme
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from urllib3.exceptions import HTTPWarning
from urllib3.util import ALPN_PROTOCOLS, resolve_cert_reqs, resolve_ssl_version
def ssl_options_to_context(keyfile=None, certfile=None, server_side=None, cert_reqs=None, ssl_version=None, ca_certs=None, do_handshake_on_connect=None, suppress_ragged_eofs=None, ciphers=None, alpn_protocols=None):
    """Return an equivalent SSLContext based on ssl.wrap_socket args."""
    ssl_version = resolve_ssl_version(ssl_version)
    cert_none = resolve_cert_reqs('CERT_NONE')
    if cert_reqs is None:
        cert_reqs = cert_none
    else:
        cert_reqs = resolve_cert_reqs(cert_reqs)
    ctx = ssl.SSLContext(ssl_version)
    ctx.load_cert_chain(certfile, keyfile)
    ctx.verify_mode = cert_reqs
    if ctx.verify_mode != cert_none:
        ctx.load_verify_locations(cafile=ca_certs)
    if alpn_protocols and hasattr(ctx, 'set_alpn_protocols'):
        try:
            ctx.set_alpn_protocols(alpn_protocols)
        except NotImplementedError:
            pass
    return ctx