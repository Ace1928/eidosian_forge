from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def get_network_extensions(self):
    """Get Cloud provided network extensions

        :returns: A set of Neutron extension aliases.
        """
    return self._neutron_extensions()