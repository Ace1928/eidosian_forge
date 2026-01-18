import socket
import io
import re
import email.utils
import email.message
import email.generator
import base64
import hmac
import copy
import datetime
import sys
from email.base64mime import body_encode as encode_base64
def auth_cram_md5(self, challenge=None):
    """ Authobject to use with CRAM-MD5 authentication. Requires self.user
        and self.password to be set."""
    if challenge is None:
        return None
    return self.user + ' ' + hmac.HMAC(self.password.encode('ascii'), challenge, 'md5').hexdigest()