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
def create_type(self, **attrs):
    """Create a new type from attributes

        :param dict attrs: Keyword arguments which will be used to create
            a :class:`~openstack.block_storage.v2.type.Type`,
            comprised of the properties on the Type class.

        :returns: The results of type creation
        :rtype: :class:`~openstack.block_storage.v2.type.Type`
        """
    return self._create(_type.Type, **attrs)