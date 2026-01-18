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
def get_torrent_file(self, fp, headers=None, cb=None, num_cb=10):
    """
        Get a torrent file (see to get_file)

        :type fp: file
        :param fp: The file pointer of where to put the torrent

        :type headers: dict
        :param headers: Headers to be passed

        :type cb: function
        :param cb: a callback function that will be called to report
            progress on the upload.  The callback should accept two
            integer parameters, the first representing the number of
            bytes that have been successfully transmitted to S3 and
            the second representing the size of the to be transmitted
            object.

        :type cb: int
        :param num_cb: (optional) If a callback is specified with the
            cb parameter this parameter determines the granularity of
            the callback by defining the maximum number of times the
            callback will be called during the file transfer.

        """
    return self.get_file(fp, headers, cb, num_cb, torrent=True)