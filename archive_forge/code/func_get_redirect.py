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
def get_redirect(self):
    """Return the redirect location configured for this key.

        If no redirect is configured (via set_redirect), then None
        will be returned.

        """
    response = self.bucket.connection.make_request('HEAD', self.bucket.name, self.name)
    if response.status == 200:
        return response.getheader('x-amz-website-redirect-location')
    else:
        raise self.provider.storage_response_error(response.status, response.reason, response.read())