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
def migrate_volume(self, volume, host=None, force_host_copy=False, lock_volume=False):
    """Migrates a volume to the specified host.

        :param volume: The value can be either the ID of a volume or a
            :class:`~openstack.block_storage.v2.volume.Volume` instance.
        :param str host: The target host for the volume migration. Host
            format is host@backend.
        :param bool force_host_copy: If false (the default), rely on the volume
            backend driver to perform the migration, which might be optimized.
            If true, or the volume driver fails to migrate the volume itself,
            a generic host-based migration is performed.
        :param bool lock_volume: If true, migrating an available volume will
            change its status to maintenance preventing other operations from
            being performed on the volume such as attach, detach, retype, etc.

        :returns: None
        """
    volume = self._get_resource(_volume.Volume, volume)
    volume.migrate(self, host, force_host_copy, lock_volume)