import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
def add_user(self, user, password):
    """Declare a user with an associated password.

        password can be empty, use an empty string ('') in that
        case, not None.
        """
    self.password_of[user] = password