import os
import time
import base64
import binascii
from libcloud.utils import iso8601
from libcloud.utils.py3 import parse_qs, urlparse, basestring
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
from libcloud.storage.types import ObjectDoesNotExistError
from libcloud.common.azure_arm import AzureResourceManagementConnection
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import Provider
from libcloud.storage.drivers.azure_blobs import AzureBlobsStorageDriver
def ex_get_public_ip(self, id):
    """
        Fetch information about a public IP resource.

        :param id: The complete resource path to the public IP resource.
        :type id: ``str`

        :return: The public ip object
        :rtype: :class:`.AzureIPAddress`
        """
    r = self.connection.request(id, params={'api-version': IP_API_VERSION})
    return self._to_ip_address(r.object)