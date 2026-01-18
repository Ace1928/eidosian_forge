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
def renew(self):
    """
        Renew the lease if it is older than a predefined time period
        """
    if self.lease_id is None:
        return
    headers = {'x-ms-lease-action': 'renew', 'x-ms-lease-id': self.lease_id, 'x-ms-lease-duration': '60'}
    response = self.driver.connection.request(self.object_path, headers=headers, params=self.params, method='PUT')
    if response.status != httplib.OK:
        raise LibcloudError('Unable to obtain lease', driver=self)