import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
def get_old_url(self, relpath=None):
    base = self.old_server.get_url()
    return self._adjust_url(base, relpath)