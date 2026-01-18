from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def remove_router_interface(self, router, subnet_id=None, port_id=None):
    """Detach a subnet from an internal router interface.

        At least one of subnet_id or port_id must be supplied.

        If you specify both subnet and port ID, the subnet ID must
        correspond to the subnet ID of the first IP address on the port
        specified by the port ID. Otherwise an error occurs.

        :param dict router: The dict object of the router being changed
        :param string subnet_id: The ID of the subnet to use for the interface
        :param string port_id: The ID of the port to use for the interface

        :returns: None on success
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    if not subnet_id and (not port_id):
        raise ValueError('At least one of subnet_id or port_id must be supplied.')
    self.network.remove_interface_from_router(router=router, subnet_id=subnet_id, port_id=port_id)