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
def delete_server(self, server, ignore_missing=True, force=False):
    """Delete a server

        :param server: The value can be either the ID of a server or a
            :class:`~openstack.compute.v2.server.Server` instance.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the server does not exist.
            When set to ``True``, no exception will be set when
            attempting to delete a nonexistent server
        :param bool force: When set to ``True``, the server deletion will be
            forced immediately.

        :returns: ``None``
        """
    if force:
        server = self._get_resource(_server.Server, server)
        server.force_delete(self)
    else:
        self._delete(_server.Server, server, ignore_missing=ignore_missing)