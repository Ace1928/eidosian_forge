import sys, datetime
from urllib.parse import urljoin, quote
from http.server import BaseHTTPRequestHandler
from urllib.error import HTTPError as urllib_HTTPError
from .extras.httpheader import content_type, parse_http_datetime
from .host import preferred_suffixes
def create_file_name(uri):
    """
    Create a suitable file name from an (absolute) URI. Used, eg, for the generation of a file name for a cached vocabulary file.
    """
    suri = uri.strip()
    final_uri = quote(suri, _unquotedChars)
    return final_uri.replace(' ', '_').replace('%', '_').replace('-', '_').replace('+', '_').replace('/', '_').replace('?', '_').replace(':', '_').replace('=', '_').replace('#', '_')