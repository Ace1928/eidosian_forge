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
def share_groups(self, **query):
    """Lists all share groups.

        :param kwargs query: Optional query parameters to be sent to limit
            the share groups being returned.  Available parameters include:

            * status: Filters by a share group status.
            * name: The user defined name of the resource to filter resources
                by.
            * description: The user defined description text that can be used
                to filter resources.
            * project_id: The project ID of the user or service.
            * share_server_id: The UUID of the share server.
            * snapshot_id: The UUID of the share’s base snapshot to filter
                the request based on.
            * host: The host name for the back end.
            * share_network_id: The UUID of the share network to filter
                resources by.
            * share_group_type_id: The share group type ID to filter
                share groups.
            * share_group_snapshot_id: The source share group snapshot ID to
                list the share group.
            * share_types: A list of one or more share type IDs. Allows
                filtering share groups.
            * limit: The maximum number of share groups members to return.
            * offset: The offset to define start point of share or share
                group listing.
            * sort_key: The key to sort a list of shares.
            * sort_dir: The direction to sort a list of shares
            * name~: The name pattern that can be used to filter shares,
                share snapshots, share networks or share groups.
            * description~: The description pattern that can be used to
                filter shares, share snapshots, share networks or share groups.

        :returns: A generator of manila share group resources
        :rtype: :class:`~openstack.shared_file_system.v2.
            share_group.ShareGroup`
        """
    return self._list(_share_group.ShareGroup, **query)