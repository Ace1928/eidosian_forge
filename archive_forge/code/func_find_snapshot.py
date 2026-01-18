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
def find_snapshot(self, name_or_id, ignore_missing=True, *, details=True, all_projects=False):
    """Find a single snapshot

        :param snapshot: The name or ID a snapshot
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be raised
            when the snapshot does not exist. When set to ``True``, None will
            be returned when attempting to find a nonexistent resource.
        :param bool details: When set to ``False``, an
            :class:`~openstack.block_storage.v2.snapshot.Snapshot` object will
            be returned. The default, ``True``, will cause an
            :class:`~openstack.block_storage.v2.snapshot.SnapshotDetail` object
            to be returned.
        :param bool all_projects: When set to ``True``, search for snapshot by
            name across all projects. Note that this will likely result in
            a higher chance of duplicates. Admin-only by default.

        :returns: One :class:`~openstack.block_storage.v2.snapshot.Snapshot`,
            one :class:`~openstack.block_storage.v2.snapshot.SnapshotDetail`
            object, or None.
        :raises: :class:`~openstack.exceptions.ResourceNotFound`
            when no resource can be found.
        :raises: :class:`~openstack.exceptions.DuplicateResource` when multiple
            resources are found.
        """
    query = {}
    if all_projects:
        query['all_projects'] = True
    list_base_path = '/snapshots/detail' if details else None
    return self._find(_snapshot.Snapshot, name_or_id, ignore_missing=ignore_missing, list_base_path=list_base_path, **query)