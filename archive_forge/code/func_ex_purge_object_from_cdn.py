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
def ex_purge_object_from_cdn(self, obj, email=None):
    """
        Purge edge cache for the specified object.

        :param email: Email where a notification will be sent when the job
        completes. (optional)
        :type email: ``str``
        """
    container_name = self._encode_container_name(obj.container.name)
    object_name = self._encode_object_name(obj.name)
    headers = {'X-Purge-Email': email} if email else {}
    response = self.connection.request('/{}/{}'.format(container_name, object_name), method='DELETE', headers=headers, cdn_request=True)
    return response.status == httplib.NO_CONTENT