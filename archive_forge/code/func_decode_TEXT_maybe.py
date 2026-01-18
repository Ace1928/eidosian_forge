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
def decode_TEXT_maybe(value):
    """
    Decode the text but only if '=?' appears in it.
    """
    return decode_TEXT(value) if '=?' in value else value