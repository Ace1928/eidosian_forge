import os
import hmac
import atexit
from time import time
from hashlib import sha1
from libcloud.utils.py3 import b, httplib, urlquote, urlencode
from libcloud.common.base import Response, RawResponse
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError, MalformedResponseError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
from libcloud.common.openstack import OpenStackDriverMixin, OpenStackBaseConnection
from libcloud.common.rackspace import AUTH_URL
from libcloud.storage.providers import Provider
from io import FileIO as file
def ex_multipart_upload_object(self, file_path, container, object_name, chunk_size=33554432, extra=None, verify_hash=True):
    object_size = os.path.getsize(file_path)
    if object_size < chunk_size:
        return self.upload_object(file_path, container, object_name, extra=extra, verify_hash=verify_hash)
    iter_chunk_reader = FileChunkReader(file_path, chunk_size)
    for index, iterator in enumerate(iter_chunk_reader):
        self._upload_object_part(container=container, object_name=object_name, part_number=index, iterator=iterator, verify_hash=verify_hash)
    return self._upload_object_manifest(container=container, object_name=object_name, extra=extra, verify_hash=verify_hash)