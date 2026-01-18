import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def ex_set_volume_labels(self, volume, labels):
    """
        Set labels for the specified volume (disk).

        :keyword  volume: The existing target StorageVolume for the request.
        :type     volume: ``StorageVolume``

        :keyword  labels: Set (or clear with None) labels for this image.
        :type     labels: ``dict`` or ``None``

        :return: True if successful
        :rtype:  ``bool``
        """
    if not isinstance(volume, StorageVolume):
        raise ValueError('Must specify a valid libcloud volume object.')
    volume_name = volume.name
    zone_name = volume.extra['zone'].name
    current_fp = volume.extra['labelFingerprint']
    body = {'labels': labels, 'labelFingerprint': current_fp}
    request = '/zones/{}/disks/{}/setLabels'.format(zone_name, volume_name)
    self.connection.async_request(request, method='POST', data=body)
    return True