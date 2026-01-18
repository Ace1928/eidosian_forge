from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_get_current_network_domain(self):
    """
        Get the current network domain ID of the driver.

        :return: ID of the network domain
        :rtype: ``str``
        """
    return self.network_domain_id