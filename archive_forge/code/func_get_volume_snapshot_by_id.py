import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
def get_volume_snapshot_by_id(self, snapshot_id):
    """Takes a snapshot_id and gets a dict of the snapshot
        that maches that ID.

        Note: This is more efficient than get_volume_snapshot.

        param: snapshot_id: ID of the volume snapshot.
        :returns: A volume ``Snapshot`` object if found, else None.
        """
    return self.block_storage.get_snapshot(snapshot_id)