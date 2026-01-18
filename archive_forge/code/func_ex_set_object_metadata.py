import os
import hmac
import base64
import hashlib
import binascii
from datetime import datetime, timedelta
from libcloud.utils.py3 import ET, b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import fixxpath
from libcloud.utils.files import read_in_chunks
from libcloud.common.azure import AzureConnection, AzureActiveDirectoryConnection
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
def ex_set_object_metadata(self, obj, meta_data):
    """
        Set metadata for an object

        :param obj: The blob object
        :type obj: :class:`Object`

        :param meta_data: Metadata key value pairs
        :type meta_data: ``dict``
        """
    object_path = self._get_object_path(obj.container, obj.name)
    params = {'comp': 'metadata'}
    headers = {}
    self._update_metadata(headers, meta_data)
    response = self.connection.request(object_path, method='PUT', params=params, headers=headers)
    if response.status != httplib.OK:
        response.parse_error('Setting metadata')