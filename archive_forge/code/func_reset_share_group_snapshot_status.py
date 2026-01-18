from openstack import exceptions
from openstack import proxy
from openstack import resource
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import limit as _limit
from openstack.shared_file_system.v2 import resource_locks as _resource_locks
from openstack.shared_file_system.v2 import share as _share
from openstack.shared_file_system.v2 import share_group as _share_group
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import share_instance as _share_instance
from openstack.shared_file_system.v2 import share_network as _share_network
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import share_snapshot as _share_snapshot
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import storage_pool as _storage_pool
from openstack.shared_file_system.v2 import user_message as _user_message
def reset_share_group_snapshot_status(self, group_snapshot_id, status):
    """Reset share group snapshot state.

        :param group_snapshot_id: The ID of the share group snapshot to reset
        :param status: The state of the share group snapshot to be set, A
            valid value is "creating", "error", "available", "deleting",
            "error_deleting".
        :rtype: ``None``
        """
    res = self._get(_share_group_snapshot.ShareGroupSnapshot, group_snapshot_id)
    res.reset_status(self, status)