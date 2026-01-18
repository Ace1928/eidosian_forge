import errno
import logging
import os
import platform
import socket
import ssl
import sys
import warnings
import pytest
from urllib3 import util
from urllib3.exceptions import HTTPWarning
from urllib3.packages import six
from urllib3.util import ssl_
def has_alpn(ctx_cls=None):
    """Detect if ALPN support is enabled."""
    ctx_cls = ctx_cls or util.SSLContext
    ctx = ctx_cls(protocol=ssl_.PROTOCOL_TLS)
    try:
        if hasattr(ctx, 'set_alpn_protocols'):
            ctx.set_alpn_protocols(ssl_.ALPN_PROTOCOLS)
            return True
    except NotImplementedError:
        pass
    return False