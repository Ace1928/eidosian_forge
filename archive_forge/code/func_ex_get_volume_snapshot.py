from libcloud.utils.py3 import httplib
from libcloud.common.ovh import API_ROOT, OvhConnection
from libcloud.compute.base import (
from libcloud.compute.types import Provider, StorageVolumeState, VolumeSnapshotState
from libcloud.compute.drivers.openstack import OpenStackKeyPair, OpenStackNodeDriver
def ex_get_volume_snapshot(self, snapshot_id):
    """
        Returns a single volume snapshot.

        :param snapshot_id: Node to run the task on.
        :type snapshot_id: ``str``

        :rtype :class:`.VolumeSnapshot`:
        :return: Volume snapshot.
        """
    action = self._get_project_action('volume/snapshot/%s' % snapshot_id)
    response = self.connection.request(action)
    return self._to_snapshot(response.object)