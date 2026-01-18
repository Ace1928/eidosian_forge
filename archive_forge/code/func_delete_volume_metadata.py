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
def delete_volume_metadata(self, volume, keys=None):
    """Delete metadata for a volume

        :param volume: Either the ID of a volume or a
            :class:`~openstack.block_storage.v2.volume.Volume`.
        :param list keys: The keys to delete. If left empty complete
            metadata will be removed.

        :rtype: ``None``
        """
    volume = self._get_resource(_volume.Volume, volume)
    if keys is not None:
        for key in keys:
            volume.delete_metadata_item(self, key)
    else:
        volume.delete_metadata(self)