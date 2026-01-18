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
def download_object_range_as_stream(self, obj, start_bytes, end_bytes=None, chunk_size=None):
    self._validate_start_and_end_bytes(start_bytes=start_bytes, end_bytes=end_bytes)
    obj_path = self._get_object_path(obj.container, obj.name)
    headers = {'Range': self._get_standard_range_str(start_bytes, end_bytes)}
    response = self.connection.request(obj_path, method='GET', headers=headers, stream=True, raw=True)
    return self._get_object(obj=obj, callback=read_in_chunks, response=response, callback_kwargs={'iterator': response.iter_content(CHUNK_SIZE), 'chunk_size': chunk_size}, success_status_code=httplib.PARTIAL_CONTENT)