from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_get_pools(self, ex_network_domain_id=None):
    """
        Get all of the pools inside the current geography or
        in given network.

        :param ex_network_domain_id: UUID of Network Domain
               if not None returns only balancers in the given network
               if None then returns all pools for the organization
        :type  ex_network_domain_id: ``str``

        :return: Returns a ``list`` of type ``NttCisPool``
        :rtype: ``list`` of ``NttCisPool``
        """
    params = None
    if ex_network_domain_id is not None:
        params = {'networkDomainId': ex_network_domain_id}
    pools = self.connection.request_with_orgId_api_2('networkDomainVip/pool', params=params).object
    return self._to_pools(pools)