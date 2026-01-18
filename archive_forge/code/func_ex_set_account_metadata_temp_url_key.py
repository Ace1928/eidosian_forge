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
def ex_set_account_metadata_temp_url_key(self, key):
    """
        Set the metadata header X-Account-Meta-Temp-URL-Key on your Cloud
        Files account.

        :param key: X-Account-Meta-Temp-URL-Key
        :type key: ``str``

        :rtype: ``bool``
        """
    headers = {'X-Account-Meta-Temp-URL-Key': key}
    response = self.connection.request('', method='POST', headers=headers, cdn_request=False)
    return response.status in [httplib.OK, httplib.NO_CONTENT, httplib.CREATED, httplib.ACCEPTED]