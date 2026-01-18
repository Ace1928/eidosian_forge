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
def download_object_as_stream(self, obj, chunk_size=None):
    obj_path = self._get_object_path(obj.container, obj.name)
    response = self.connection.request(obj_path, method='GET', stream=True, raw=True)
    return self._get_object(obj=obj, callback=read_in_chunks, response=response, callback_kwargs={'iterator': response.iter_content(CHUNK_SIZE), 'chunk_size': chunk_size}, success_status_code=httplib.OK)