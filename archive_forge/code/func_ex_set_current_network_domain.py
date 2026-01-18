from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_set_current_network_domain(self, network_domain_id):
    """
        Set the network domain (part of the network) of the driver

        :param network_domain_id: ID of the pool (required)
        :type  network_domain_id: ``str``
        """
    self.network_domain_id = network_domain_id