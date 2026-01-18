from keystoneauth1 import exceptions as ksa_exceptions
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.i18n import _
def network_delete(self, network=None):
    """Delete a network

        https://docs.openstack.org/api-ref/compute/#delete-network

        :param string network:
            Network name or ID
        """
    url = '/os-networks'
    network = self.find(url, attr='label', value=network)['id']
    if network is not None:
        return self.delete('/%s/%s' % (url, network))
    return None