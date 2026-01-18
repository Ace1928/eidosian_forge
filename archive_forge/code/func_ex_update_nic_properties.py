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
def ex_update_nic_properties(self, network_interface, resource_group, properties):
    """
        Update the properties of an already existing virtual network
        interface (NIC).

        :param network_interface: The NIC to update.
        :type network_interface: :class:`.AzureNic`

        :param resource_group: The resource group to check the ip address in.
        :type resource_group: ``str``

        :param properties: The dictionary of the NIC's properties
        :type properties: ``dict``

        :return: The NIC object
        :rtype: :class:`.AzureNic`
        """
    target = '/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/networkInterfaces/%s' % (self.subscription_id, resource_group, network_interface.name)
    data = {'properties': properties, 'location': network_interface.location}
    r = self.connection.request(target, params={'api-version': NIC_API_VERSION}, data=data, method='PUT')
    return AzureNic(r.object['id'], r.object['name'], r.object['location'], r.object['properties'])