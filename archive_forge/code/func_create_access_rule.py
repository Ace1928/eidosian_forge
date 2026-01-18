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
def create_access_rule(self, share_id, **attrs):
    """Creates an access rule from attributes

        :returns: Details of the new access rule
        :param share_id: The ID of the share
        :param dict attrs: Attributes which will be used to create
            a :class:`~openstack.shared_file_system.v2.
            share_access_rules.ShareAccessRules`, comprised of the
            properties on the ShareAccessRules class.
        :rtype: :class:`~openstack.shared_file_system.v2.
            share_access_rules.ShareAccessRules`
        """
    base_path = '/shares/%s/action' % (share_id,)
    return self._create(_share_access_rule.ShareAccessRule, base_path=base_path, **attrs)