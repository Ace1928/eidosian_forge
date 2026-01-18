from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def get_network_by_id(self, id):
    """Get a network by ID

        :param id: ID of the network.
        :returns: A network ``Network`` object if found, else None.
        """
    return self.network.get_network(id)