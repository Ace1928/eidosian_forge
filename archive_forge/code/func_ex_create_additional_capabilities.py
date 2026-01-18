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
def ex_create_additional_capabilities(self, node, additional_capabilities, resource_group):
    """
        Set the additional capabilities on a stopped node.

        :param node: The node to be updated
        :type node: :class:`.Node`

        :param ex_additional_capabilities: Optional additional capabilities
            allowing Ultra SSD and hibernation on this node.
        :type ex_additional_capabilities: ``dict``

        :param resource_group: The resource group of the node to be updated
        :type resource_group: ``str``

        :return: True if the update was successful, otherwise False
        :rtype: ``bool``
        """
    target = '/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Compute/virtualMachines/%s' % (self.subscription_id, resource_group, node.name)
    data = {'location': node.extra['location'], 'properties': {'additionalCapabilities': additional_capabilities}}
    r = self.connection.request(target, data=data, params={'api-version': VM_API_VERSION}, method='PUT')
    return r.status in [200, 202, 204]