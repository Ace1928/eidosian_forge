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
def share_group_snapshots(self, details=True, **query):
    """Lists all share group snapshots.

        :param kwargs query: Optional query parameters to be sent
            to limit the share group snapshots being returned.
            Available parameters include:

            * project_id: The ID of the project that owns the resource.
            * name: The user defined name of the resource to filter resources.
            * description: The user defined description text that can be used
              to filter resources.
            * status: Filters by a share status
            * share_group_id: The UUID of a share group to filter resource.
            * limit: The maximum number of share group snapshot members
              to return.
            * offset: The offset to define start point of share or
              share group listing.
            * sort_key: The key to sort a list of shares.
            * sort_dir: The direction to sort a list of shares. A valid
              value is asc, or desc.

        :returns: Details of share group snapshots resources
        :rtype: :class:`~openstack.shared_file_system.v2.
            share_group_snapshot.ShareGroupSnapshot`
        """
    base_path = '/share-group-snapshots/detail' if details else None
    return self._list(_share_group_snapshot.ShareGroupSnapshot, base_path=base_path, **query)