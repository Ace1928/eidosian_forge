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
def change_storage_class(self, new_storage_class, dst_bucket=None, validate_dst_bucket=True):
    """
        Change the storage class of an existing key.
        Depending on whether a different destination bucket is supplied
        or not, this will either move the item within the bucket, preserving
        all metadata and ACL info bucket changing the storage class or it
        will copy the item to the provided destination bucket, also
        preserving metadata and ACL info.

        :type new_storage_class: string
        :param new_storage_class: The new storage class for the Key.
            Possible values are:
            * STANDARD
            * REDUCED_REDUNDANCY

        :type dst_bucket: string
        :param dst_bucket: The name of a destination bucket.  If not
            provided the current bucket of the key will be used.

        :type validate_dst_bucket: bool
        :param validate_dst_bucket: If True, will validate the dst_bucket
            by using an extra list request.
        """
    bucket_name = dst_bucket or self.bucket.name
    if new_storage_class == 'STANDARD':
        return self.copy(bucket_name, self.name, reduced_redundancy=False, preserve_acl=True, validate_dst_bucket=validate_dst_bucket)
    elif new_storage_class == 'REDUCED_REDUNDANCY':
        return self.copy(bucket_name, self.name, reduced_redundancy=True, preserve_acl=True, validate_dst_bucket=validate_dst_bucket)
    else:
        raise BotoClientError('Invalid storage class: %s' % new_storage_class)