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
def set_volume_metadata(self, volume, **metadata):
    """Update metadata for a volume

        :param volume: Either the ID of a volume or a
            :class:`~openstack.block_storage.v2.volume.Volume`.
        :param kwargs metadata: Key/value pairs to be updated in the volume's
            metadata. No other metadata is modified by this call. All keys
            and values are stored as Unicode.

        :returns: A :class:`~openstack.block_storage.v2.volume.Volume` with the
            volume's metadata. All keys and values are Unicode text.
        :rtype: :class:`~openstack.block_storage.v2.volume.Volume`
        """
    volume = self._get_resource(_volume.Volume, volume)
    return volume.set_metadata(self, metadata=metadata)