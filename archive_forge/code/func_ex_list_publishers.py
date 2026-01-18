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
def ex_list_publishers(self, location=None):
    """
        List node image publishers.

        :param location: The location at which to list publishers
        (if None, use default location specified as 'region' in __init__)
        :type location: :class:`.NodeLocation`

        :return: A list of tuples in the form
        ("publisher id", "publisher name")
        :rtype: ``list``
        """
    if location is None:
        if self.default_location:
            location = self.default_location
        else:
            raise ValueError('location is required.')
    action = '/subscriptions/%s/providers/Microsoft.Compute/locations/%s/publishers' % (self.subscription_id, location.id)
    r = self.connection.request(action, params={'api-version': IMAGES_API_VERSION})
    return [(p['id'], p['name']) for p in r.object]