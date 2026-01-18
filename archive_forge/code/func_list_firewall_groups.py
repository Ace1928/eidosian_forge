from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def list_firewall_groups(self, filters=None):
    """
        Lists firewall groups.

        :returns: A list of network ``FirewallGroup`` objects.
        """
    if not filters:
        filters = {}
    return list(self.network.firewall_groups(**filters))