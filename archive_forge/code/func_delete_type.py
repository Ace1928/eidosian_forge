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
def delete_type(self, type, ignore_missing=True):
    """Delete a type

        :param type: The value can be either the ID of a type or a
            :class:`~openstack.block_storage.v2.type.Type` instance.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the type does not exist.
            When set to ``True``, no exception will be set when
            attempting to delete a nonexistent type.

        :returns: ``None``
        """
    self._delete(_type.Type, type, ignore_missing=ignore_missing)