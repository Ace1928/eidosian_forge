from keystoneauth1 import exceptions as ksa_exceptions
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.i18n import _
def network_create(self, name=None, subnet=None, share_subnet=None):
    """Create a new network

        https://docs.openstack.org/api-ref/compute/#create-network

        :param string name:
            Network label (required)
        :param integer subnet:
            Subnet for IPv4 fixed addresses in CIDR notation (required)
        :param integer share_subnet:
            Shared subnet between projects, True or False
        :returns: A dict of the network attributes
        """
    url = '/os-networks'
    params = {'label': name, 'cidr': subnet}
    if share_subnet is not None:
        params['share_address'] = share_subnet
    return self.create(url, json={'network': params})['network']