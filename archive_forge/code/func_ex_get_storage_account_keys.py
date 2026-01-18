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
def ex_get_storage_account_keys(self, resource_group, storage_account):
    """
        Get account keys required to access to a storage account
        (using AzureBlobsStorageDriver).

        :param resource_group: The resource group
            containing the storage account
        :type resource_group: ``str``

        :param storage_account: Storage account to access
        :type storage_account: ``str``

        :return: The account keys, in the form `{"key1": "XXX", "key2": "YYY"}`
        :rtype: ``.dict``
        """
    action = '/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Storage/storageAccounts/%s/listKeys' % (self.subscription_id, resource_group, storage_account)
    r = self.connection.request(action, params={'api-version': STORAGE_ACCOUNT_API_VERSION}, method='POST')
    return r.object