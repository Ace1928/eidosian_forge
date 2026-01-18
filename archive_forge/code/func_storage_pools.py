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
def storage_pools(self, details=True, **query):
    """Lists all back-end storage pools with details

        :param kwargs query: Optional query parameters to be sent to limit
            the storage pools being returned.  Available parameters include:

            * pool_name: The pool name for the back end.
            * host_name: The host name for the back end.
            * backend_name: The name of the back end.
            * capabilities: The capabilities for the storage back end.
            * share_type: The share type name or UUID.
        :returns: A generator of manila storage pool resources
        :rtype:
            :class:`~openstack.shared_file_system.v2.storage_pool.StoragePool`
        """
    base_path = '/scheduler-stats/pools/detail' if details else None
    return self._list(_storage_pool.StoragePool, base_path=base_path, **query)