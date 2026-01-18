import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
def send_header_auth_reqed(self):
    tcs = self.server.test_case_server
    self.send_header(tcs.auth_header_sent, 'Basic realm="%s"' % tcs.auth_realm)
    header = 'Digest realm="%s", ' % tcs.auth_realm
    header += 'nonce="{}", algorithm="{}", qop="auth"'.format(tcs.auth_nonce, 'MD5')
    self.send_header(tcs.auth_header_sent, header)