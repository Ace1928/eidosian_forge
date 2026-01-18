from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_get_default_persistence_profiles(self, network_domain_id):
    """
        Get the default persistence profiles available for a network domain

        :param network_domain_id: The ID of of a ``NttCisNetworkDomain``
        :type  network_domain_id: ``str``

        :rtype: `list` of :class:`NttCisPersistenceProfile`
        """
    result = self.connection.request_with_orgId_api_2(action='networkDomainVip/defaultPersistenceProfile', params={'networkDomainId': network_domain_id}, method='GET').object
    return self._to_persistence_profiles(result)