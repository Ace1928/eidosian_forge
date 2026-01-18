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
def ex_check_ip_address_availability(self, resource_group, network, ip_address):
    """
        Checks whether a private IP address is available for use. Also
        returns an object that contains the available IPs in the subnet.

        :param resource_group: The resource group to check the ip address in.
        :type resource_group: ``str``

        :param network: The virtual network.
        :type network: :class:`.AzureNetwork`

        :param ip_address: The private IP address to be verified.
        :type ip_address: ``str``
        """
    params = {'api-version': VIRTUAL_NETWORK_API_VERSION}
    action = '/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/virtualNetworks/%s/CheckIPAddressAvailability' % (self.subscription_id, resource_group, network.name)
    if ip_address is not None:
        params['ipAddress'] = ip_address
    r = self.connection.request(action, params=params)
    return r.object