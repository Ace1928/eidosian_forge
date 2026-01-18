from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def get_subnet_by_id(self, id):
    """Get a subnet by ID

        :param id: ID of the subnet.
        :returns: A network ``Subnet`` object if found, else None.
        """
    return self.network.get_subnet(id)