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
def set_volume_bootable_status(self, volume, bootable):
    """Set bootable status of the volume.

        :param volume: The value can be either the ID of a volume or a
            :class:`~openstack.block_storage.v2.volume.Volume` instance.
        :param bool bootable: Specifies whether the volume should be bootable
            or not.

        :returns: None
        """
    volume = self._get_resource(_volume.Volume, volume)
    volume.set_bootable_status(self, bootable)