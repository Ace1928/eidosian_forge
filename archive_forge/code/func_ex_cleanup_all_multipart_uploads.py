import os
import hmac
import time
import base64
from typing import Dict, Optional
from hashlib import sha1
from datetime import datetime
import libcloud.utils.py3
from libcloud.utils.py3 import b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import findtext, fixxpath
from libcloud.common.aws import (
from libcloud.common.base import RawResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
def ex_cleanup_all_multipart_uploads(self, container, prefix=None):
    """
        Extension method for removing all partially completed S3 multipart
        uploads.

        :param container: The container holding the uploads
        :type container: :class:`Container`

        :keyword prefix: Delete only uploads of objects with this prefix
        :type prefix: ``str``
        """
    for upload in self.ex_iterate_multipart_uploads(container, prefix, delimiter=None):
        self._abort_multipart(container, upload.key, upload.id)