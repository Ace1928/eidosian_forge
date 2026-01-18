import threading
from openstack import exceptions
def get_external_ipv4_floating_networks(self):
    """Return the networks that are configured to route northbound.

        :returns: A list of network ``Network`` objects if any are found
        """
    self._find_interesting_networks()
    return self._external_ipv4_floating_networks