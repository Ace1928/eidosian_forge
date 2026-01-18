import base64
import binascii
import os
import re
from boto.compat import StringIO, six
from boto.exception import BotoClientError
from boto.s3.key import Key as S3Key
from boto.s3.keyfile import KeyFile
from boto.utils import compute_hash, get_utf8able_str
def handle_version_headers(self, resp, force=False):
    self.metageneration = resp.getheader('x-goog-metageneration', None)
    self.generation = resp.getheader('x-goog-generation', None)