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
def iterate_container_objects(self, container, prefix=None, ex_prefix=None):
    """
        Return a generator of objects for the given container.

        :param container: Container instance
        :type container: :class:`Container`

        :param prefix: Only return objects starting with prefix
        :type prefix: ``str``

        :param ex_prefix: Only return objects starting with ex_prefix
        :type ex_prefix: ``str``

        :return: A generator of Object instances.
        :rtype: ``generator`` of :class:`Object`
        """
    prefix = self._normalize_prefix_argument(prefix, ex_prefix)
    params = {}
    if prefix:
        params['prefix'] = prefix
    last_key = None
    exhausted = False
    container_path = self._get_container_path(container)
    while not exhausted:
        if last_key:
            params['marker'] = last_key
        response = self.connection.request(container_path, params=params)
        if response.status != httplib.OK:
            raise LibcloudError('Unexpected status code: %s' % response.status, driver=self)
        objects = self._to_objs(obj=response.object, xpath='Contents', container=container)
        is_truncated = response.object.findtext(fixxpath(xpath='IsTruncated', namespace=self.namespace)).lower()
        exhausted = is_truncated == 'false'
        last_key = None
        for obj in objects:
            last_key = obj.name
            yield obj