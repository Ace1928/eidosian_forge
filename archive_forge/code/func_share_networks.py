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
def share_networks(self, details=True, **query):
    """Lists all share networks with details.

        :param dict query: Optional query parameters to be sent to limit the
            resources being returned. Available parameters include:

            * name~: The user defined name of the resource to filter resources
              by.
            * project_id: The ID of the user or service making the request.
            * description~: The description pattern that can be used to filter
              shares, share snapshots, share networks or share groups.
            * all_projects: (Admin only). Defines whether to list the requested
              resources for all projects.

        :returns: Details of shares networks
        :rtype: :class:`~openstack.shared_file_system.v2.
            share_network.ShareNetwork`
        """
    base_path = '/share-networks/detail' if details else None
    return self._list(_share_network.ShareNetwork, base_path=base_path, **query)