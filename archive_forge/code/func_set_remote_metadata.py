from __future__ import print_function
import email.utils
import errno
import hashlib
import mimetypes
import os
import re
import base64
import binascii
import math
from hashlib import md5
import boto.utils
from boto.compat import BytesIO, six, urllib, encodebytes
from boto.exception import BotoClientError
from boto.exception import StorageDataError
from boto.exception import PleaseRetryException
from boto.exception import ResumableDownloadException
from boto.exception import ResumableTransferDisposition
from boto.provider import Provider
from boto.s3.keyfile import KeyFile
from boto.s3.user import User
from boto import UserAgent
import boto.utils
from boto.utils import compute_md5, compute_hash
from boto.utils import find_matching_headers
from boto.utils import merge_headers_by_name
from boto.utils import print_to_fd
def set_remote_metadata(self, metadata_plus, metadata_minus, preserve_acl, headers=None):
    metadata_plus = self._normalize_metadata(metadata_plus)
    metadata_minus = self._normalize_metadata(metadata_minus)
    metadata = self._get_remote_metadata()
    metadata.update(metadata_plus)
    for h in metadata_minus:
        if h in metadata:
            del metadata[h]
    src_bucket = self.bucket
    rewritten_metadata = {}
    for h in metadata:
        if h.startswith('x-goog-meta-') or h.startswith('x-amz-meta-'):
            rewritten_h = h.replace('x-goog-meta-', '').replace('x-amz-meta-', '')
        else:
            rewritten_h = h
        rewritten_metadata[rewritten_h] = metadata[h]
    metadata = rewritten_metadata
    src_bucket.copy_key(self.name, self.bucket.name, self.name, metadata=metadata, preserve_acl=preserve_acl, headers=headers)