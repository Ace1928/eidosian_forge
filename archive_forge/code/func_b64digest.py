import base64
import io
import logging
from binascii import crc32
from hashlib import sha1, sha256
from botocore.compat import HAS_CRT
from botocore.exceptions import (
from botocore.response import StreamingBody
from botocore.utils import (
def b64digest(self):
    bs = self.digest()
    return base64.b64encode(bs).decode('ascii')