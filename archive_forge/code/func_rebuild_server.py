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
def rebuild_server(self, server, image, **attrs):
    """Rebuild a server

        :param server: Either the ID of a server or a
            :class:`~openstack.compute.v2.server.Server` instance.
        :param str name: The name of the server
        :param str admin_password: The administrator password
        :param bool preserve_ephemeral: Indicates whether the server
            is rebuilt with the preservation of the ephemeral partition.
            *Default: False*
        :param str image: The id of an image to rebuild with. *Default: None*
        :param str access_ipv4: The IPv4 address to rebuild with.
            *Default: None*
        :param str access_ipv6: The IPv6 address to rebuild with.
            *Default: None*
        :param dict metadata: A dictionary of metadata to rebuild with.
            *Default: None*
        :param personality: A list of dictionaries, each including a
            **path** and **contents** key, to be injected
            into the rebuilt server at launch.
            *Default: None*

        :returns: The rebuilt :class:`~openstack.compute.v2.server.Server`
            instance.
        """
    server = self._get_resource(_server.Server, server)
    return server.rebuild(self, image=image, **attrs)