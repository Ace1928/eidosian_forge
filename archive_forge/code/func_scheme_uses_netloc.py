import re
import sys
import string
import socket
from socket import AF_INET, AF_INET6
from typing import (
from unicodedata import normalize
from ._socket import inet_pton
from idna import encode as idna_encode, decode as idna_decode
def scheme_uses_netloc(scheme, default=None):
    """Whether or not a URL uses :code:`:` or :code:`://` to separate the
    scheme from the rest of the URL depends on the scheme's own
    standard definition. There is no way to infer this behavior
    from other parts of the URL. A scheme either supports network
    locations or it does not.

    The URL type's approach to this is to check for explicitly
    registered schemes, with common schemes like HTTP
    preregistered. This is the same approach taken by
    :mod:`urlparse`.

    URL adds two additional heuristics if the scheme as a whole is
    not registered. First, it attempts to check the subpart of the
    scheme after the last ``+`` character. This adds intuitive
    behavior for schemes like ``git+ssh``. Second, if a URL with
    an unrecognized scheme is loaded, it will maintain the
    separator it sees.
    """
    if not scheme:
        return False
    scheme = scheme.lower()
    if scheme in SCHEME_PORT_MAP:
        return True
    if scheme in NO_NETLOC_SCHEMES:
        return False
    if scheme.split('+')[-1] in SCHEME_PORT_MAP:
        return True
    return default