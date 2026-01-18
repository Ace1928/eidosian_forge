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
def get_export_location(self, export_location, share_id):
    """List details of export location

        :param export_location: The export location resource to get
        :param share_id: The ID of the share to get export locations from
        :returns: Details of identified export location
        :rtype: :class:`~openstack.shared_filesystem_storage.v2.
            share_export_locations.ShareExportLocations`
        """
    export_location_id = resource.Resource._get_id(export_location)
    return self._get(_share_export_locations.ShareExportLocation, export_location_id, share_id=share_id)