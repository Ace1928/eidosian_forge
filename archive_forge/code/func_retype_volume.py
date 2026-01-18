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
def retype_volume(self, volume, new_type, migration_policy='never'):
    """Retype the volume.

        :param volume: The value can be either the ID of a volume or a
            :class:`~openstack.block_storage.v2.volume.Volume` instance.
        :param str new_type: The new volume type that volume is changed with.
        :param str migration_policy: Specify if the volume should be migrated
            when it is re-typed. Possible values are on-demand or never.
            Default: never.

        :returns: None
        """
    volume = self._get_resource(_volume.Volume, volume)
    volume.retype(self, new_type, migration_policy)