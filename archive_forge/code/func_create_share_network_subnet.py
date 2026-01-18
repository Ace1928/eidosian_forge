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
def create_share_network_subnet(self, share_network_id, **attrs):
    """Creates a share network subnet from attributes

        :param share_network_id: The id of the share network wthin which the
            the Share Network Subnet should be created.
        :param dict attrs: Attributes which will be used to create
            a share network subnet.
        :returns: Details of the new share network subnet.
        :rtype:
            :class:`~openstack.shared_file_system.v2.share_network_subnet.ShareNetworkSubnet`
        """
    return self._create(_share_network_subnet.ShareNetworkSubnet, **attrs, share_network_id=share_network_id)