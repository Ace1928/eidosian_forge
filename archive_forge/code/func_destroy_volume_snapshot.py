import copy
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState, VolumeSnapshotState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def destroy_volume_snapshot(self, snapshot, region=None):
    """
        Dostroy a volume snapshot

        :param snapshot: volume snapshot to destroy
        :type snapshot: class:`VolumeSnapshot`

        :param region: The region in which to look for the snapshot
        (if None, use default region specified in __init__)
        :type region: :class:`.NodeLocation`

        :return: True if the destroy was successful, otherwise False
        :rtype: ``bool``
        """
    return self.connection.request('/snapshots/%s' % snapshot.id, region=region, method='DELETE').success()