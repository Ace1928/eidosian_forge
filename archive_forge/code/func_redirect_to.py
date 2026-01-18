import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
def redirect_to(self, host, port):
    """Redirect all requests to a specific host:port"""
    self.redirections = [('(.*)', 'http://{}:{}\\1'.format(host, port), 301)]