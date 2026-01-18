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
def parse_query_string(query_string, keep_blank_values=True, encoding='utf-8'):
    """Build a params dictionary from a query_string.

    Duplicate key/value pairs in the provided query_string will be
    returned as {'key': [val1, val2, ...]}. Single key/values will
    be returned as strings: {'key': 'value'}.
    """
    if image_map_pattern.match(query_string):
        pm = query_string.split(',')
        pm = {'x': int(pm[0]), 'y': int(pm[1])}
    else:
        pm = _parse_qs(query_string, keep_blank_values, encoding=encoding)
    return pm