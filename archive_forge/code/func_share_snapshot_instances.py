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
def share_snapshot_instances(self, details=True, **query):
    """Lists all share snapshot instances with details.

        :param bool details: Whether to fetch detailed resource
            descriptions. Defaults to True.
        :param kwargs query: Optional query parameters to be sent to limit
            the share snapshot instance being returned.
            Available parameters include:

            * snapshot_id: The UUID of the share’s base snapshot to filter
                the request based on.
            * project_id: The project ID of the user or service making the
                request.

        :returns: A generator of share snapshot instance resources
        :rtype: :class:`~openstack.shared_file_system.v2.
            share_snapshot_instance.ShareSnapshotInstance`
        """
    base_path = '/snapshot-instances/detail' if details else None
    return self._list(_share_snapshot_instance.ShareSnapshotInstance, base_path=base_path, **query)