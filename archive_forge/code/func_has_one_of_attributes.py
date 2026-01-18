import sys, datetime
from urllib.parse import urljoin, quote
from http.server import BaseHTTPRequestHandler
from urllib.error import HTTPError as urllib_HTTPError
from .extras.httpheader import content_type, parse_http_datetime
from .host import preferred_suffixes
def has_one_of_attributes(node, *args):
    """
    Check whether one of the listed attributes is present on a (DOM) node.
    @param node: DOM element node
    @param args: possible attribute names
    @return: True or False
    @rtype: Boolean
    """
    if len(args) == 0:
        return None
    if isinstance(args[0], tuple) or isinstance(args[0], list):
        rargs = args[0]
    else:
        rargs = args
    return True in [node.hasAttribute(attr) for attr in rargs]