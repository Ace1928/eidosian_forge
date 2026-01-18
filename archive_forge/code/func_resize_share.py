from openstack import exceptions
from openstack import proxy
from openstack import resource
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import limit as _limit
from openstack.shared_file_system.v2 import resource_locks as _resource_locks
from openstack.shared_file_system.v2 import share as _share
from openstack.shared_file_system.v2 import share_group as _share_group
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import share_instance as _share_instance
from openstack.shared_file_system.v2 import share_network as _share_network
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import share_snapshot as _share_snapshot
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import storage_pool as _storage_pool
from openstack.shared_file_system.v2 import user_message as _user_message
def resize_share(self, share_id, new_size, no_shrink=False, no_extend=False, force=False):
    """Resizes a share, extending/shrinking the share as needed.

        :param share_id: The ID of the share to resize
        :param new_size: The new size of the share in GiBs. If new_size is
            the same as the current size, then nothing is done.
        :param bool no_shrink: If set to True, the given share is not shrunk,
            even if shrinking the share is required to get the share to the
            given size. This could be useful for extending shares to a minimum
            size, while not shrinking shares to the given size. This defaults
            to False.
        :param bool no_extend: If set to True, the given share is not
            extended, even if extending the share is required to get the share
            to the given size. This could be useful for shrinking shares to a
            maximum size, while not extending smaller shares to that maximum
            size. This defaults to False.
        :param bool force: Whether or not force should be used,
            in the case where the share should be extended.
        :returns: ``None``
        """
    res = self._get(_share.Share, share_id)
    if new_size > res.size and no_extend is not True:
        res.extend_share(self, new_size, force)
    elif new_size < res.size and no_shrink is not True:
        res.shrink_share(self, new_size)