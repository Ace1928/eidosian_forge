import errno
import os
import random
import re
import socket
import time
from hashlib import md5
import six.moves.http_client as httplib
from six.moves import urllib as urlparse
from boto import config, UserAgent
from boto.connection import AWSAuthConnection
from boto.exception import InvalidUriError
from boto.exception import ResumableTransferDisposition
from boto.exception import ResumableUploadException
from boto.s3.keyfile import KeyFile
def get_upload_id(self):
    """
        Returns the upload ID for the resumable upload, or None if the upload
        has not yet started.
        """
    delim = '?upload_id='
    if self.tracker_uri and delim in self.tracker_uri:
        return self.tracker_uri[self.tracker_uri.index(delim) + len(delim):]
    else:
        return None