import logging
import ssl
import sys
from . import __author__, __copyright__, __license__, __version__, TIMEOUT
from .simplexml import SimpleXMLElement, TYPE_MAP, Struct
def set_http_wrapper(library=None, features=[]):
    """Set a suitable HTTP connection wrapper."""
    global Http
    Http = get_http_wrapper(library, features)
    return Http