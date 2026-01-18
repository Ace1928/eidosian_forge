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
def get_md5_from_hexdigest(self, md5_hexdigest):
    """
        A utility function to create the 2-tuple (md5hexdigest, base64md5)
        from just having a precalculated md5_hexdigest.
        """
    digest = binascii.unhexlify(md5_hexdigest)
    base64md5 = encodebytes(digest)
    if base64md5[-1] == '\n':
        base64md5 = base64md5[0:-1]
    return (md5_hexdigest, base64md5)