import functools
import email.utils
import re
import builtins
from binascii import b2a_base64
from cgi import parse_header
from email.header import decode_header
from http.server import BaseHTTPRequestHandler
from urllib.parse import unquote_plus
import jaraco.collections
import cherrypy
from cherrypy._cpcompat import ntob, ntou
@classmethod
def encode_header_items(cls, header_items):
    """
        Prepare the sequence of name, value tuples into a form suitable for
        transmitting on the wire for HTTP.
        """
    for k, v in header_items:
        if not isinstance(v, str) and (not isinstance(v, bytes)):
            v = str(v)
        yield tuple(map(cls.encode_header_item, (k, v)))