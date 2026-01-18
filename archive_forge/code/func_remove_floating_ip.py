import typing as ty
from openstack.common import metadata
from openstack.common import tag
from openstack.compute.v2 import flavor
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.image.v2 import image
from openstack import resource
from openstack import utils
def remove_floating_ip(self, session, address):
    """Remove a floating IP from the server.

        :param session: The session to use for making this request.
        :param address: The floating IP address to disassociate from the
            server.
        :returns: None
        """
    body = {'removeFloatingIp': {'address': address}}
    self._action(session, body)