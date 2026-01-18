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
def get_share_snapshot_instance(self, snapshot_instance_id):
    """Lists details of a single share snapshot instance

        :param snapshot_instance_id: The ID of the snapshot instance to get
        :returns: Details of the identified snapshot instance
        :rtype: :class:`~openstack.shared_file_system.v2.
            share_snapshot_instance.ShareSnapshotInstance`
        """
    return self._get(_share_snapshot_instance.ShareSnapshotInstance, snapshot_instance_id)