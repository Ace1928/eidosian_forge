from openstack.block_storage import _base_proxy
from openstack.block_storage.v2 import backup as _backup
from openstack.block_storage.v2 import capabilities as _capabilities
from openstack.block_storage.v2 import extension as _extension
from openstack.block_storage.v2 import limits as _limits
from openstack.block_storage.v2 import quota_set as _quota_set
from openstack.block_storage.v2 import snapshot as _snapshot
from openstack.block_storage.v2 import stats as _stats
from openstack.block_storage.v2 import type as _type
from openstack.block_storage.v2 import volume as _volume
from openstack.identity.v3 import project as _project
from openstack import resource
def get_type_access(self, type):
    """Lists project IDs that have access to private volume type.

        :param type: The value can be either the ID of a type or a
            :class:`~openstack.block_storage.v2.type.Type` instance.

        :returns: List of dictionaries describing projects that have access to
            the specified type
        """
    res = self._get_resource(_type.Type, type)
    return res.get_private_access(self)