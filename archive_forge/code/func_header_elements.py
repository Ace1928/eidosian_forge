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
def header_elements(fieldname, fieldvalue):
    """Return a sorted HeaderElement list from a comma-separated header string.
    """
    if not fieldvalue:
        return []
    result = []
    for element in RE_HEADER_SPLIT.split(fieldvalue):
        if fieldname.startswith('Accept') or fieldname == 'TE':
            hv = AcceptElement.from_str(element)
        else:
            hv = HeaderElement.from_str(element)
        result.append(hv)
    return list(reversed(sorted(result)))