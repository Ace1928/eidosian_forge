import warnings
from openstack.block_storage.v3 import volume as _volume
from openstack.compute.v2 import aggregate as _aggregate
from openstack.compute.v2 import availability_zone
from openstack.compute.v2 import extension
from openstack.compute.v2 import flavor as _flavor
from openstack.compute.v2 import hypervisor as _hypervisor
from openstack.compute.v2 import image as _image
from openstack.compute.v2 import keypair as _keypair
from openstack.compute.v2 import limits
from openstack.compute.v2 import migration as _migration
from openstack.compute.v2 import quota_set as _quota_set
from openstack.compute.v2 import server as _server
from openstack.compute.v2 import server_action as _server_action
from openstack.compute.v2 import server_diagnostics as _server_diagnostics
from openstack.compute.v2 import server_group as _server_group
from openstack.compute.v2 import server_interface as _server_interface
from openstack.compute.v2 import server_ip
from openstack.compute.v2 import server_migration as _server_migration
from openstack.compute.v2 import server_remote_console as _src
from openstack.compute.v2 import service as _service
from openstack.compute.v2 import usage as _usage
from openstack.compute.v2 import volume_attachment as _volume_attachment
from openstack import exceptions
from openstack.identity.v3 import project as _project
from openstack.network.v2 import security_group as _sg
from openstack import proxy
from openstack import resource
from openstack import utils
from openstack import warnings as os_warnings
def find_server(self, name_or_id, ignore_missing=True, *, details=True, all_projects=False):
    """Find a single server

        :param name_or_id: The name or ID of a server.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be raised when
            the resource does not exist. When set to ``True``, None will be
            returned when attempting to find a nonexistent resource.
        :param bool details: When set to ``False``
            instances with only basic data will be returned. The default,
            ``True``, will cause instances with full data to be returned.
        :param bool all_projects: When set to ``True``, search for server
            by name across all projects. Note that this will likely result in a
            higher chance of duplicates. Admin-only by default.

        :returns: One :class:`~openstack.compute.v2.server.Server` or None
        :raises: :class:`~openstack.exceptions.ResourceNotFound`
            when no resource can be found.
        :raises: :class:`~openstack.exceptions.DuplicateResource` when multiple
            resources are found.
        """
    query = {}
    if all_projects:
        query['all_projects'] = True
    list_base_path = '/servers/detail' if details else None
    return self._find(_server.Server, name_or_id, ignore_missing=ignore_missing, list_base_path=list_base_path, **query)