import base64
import binascii
import os
import re
from boto.compat import StringIO, six
from boto.exception import BotoClientError
from boto.s3.key import Key as S3Key
from boto.s3.keyfile import KeyFile
from boto.utils import compute_hash, get_utf8able_str
def handle_addl_headers(self, headers):
    for key, value in headers:
        if key == 'x-goog-hash':
            for hash_pair in value.split(','):
                alg, b64_digest = hash_pair.strip().split('=', 1)
                self.cloud_hashes[alg] = binascii.a2b_base64(b64_digest)
        elif key == 'x-goog-component-count':
            self.component_count = int(value)
        elif key == 'x-goog-generation':
            self.generation = value
        elif key == 'x-goog-stored-content-encoding':
            self.content_encoding = value
        elif key == 'x-goog-stored-content-length':
            self.size = int(value)
        elif key == 'x-goog-storage-class':
            self.storage_class = value