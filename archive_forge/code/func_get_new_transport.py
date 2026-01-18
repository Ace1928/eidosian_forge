import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
def get_new_transport(self, relpath=None):
    t = transport.get_transport_from_url(self.get_new_url(relpath))
    self.assertTrue(t.is_readonly())
    return t