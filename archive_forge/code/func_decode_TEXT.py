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
def decode_TEXT(value):
    """
    Decode :rfc:`2047` TEXT

    >>> decode_TEXT("=?utf-8?q?f=C3=BCr?=") == b'f\\xfcr'.decode('latin-1')
    True
    """
    atoms = decode_header(value)
    decodedvalue = ''
    for atom, charset in atoms:
        if charset is not None:
            atom = atom.decode(charset)
        decodedvalue += atom
    return decodedvalue