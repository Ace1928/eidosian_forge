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
def reset_snapshot(self, snapshot, status):
    """Reset status of the snapshot

        :param snapshot: The value can be either the ID of a backup or a
            :class:`~openstack.block_storage.v2.snapshot.Snapshot` instance.
        :param str status: New snapshot status

        :returns: None
        """
    snapshot = self._get_resource(_snapshot.Snapshot, snapshot)
    snapshot.reset(self, status)