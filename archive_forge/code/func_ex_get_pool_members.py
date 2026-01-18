from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_get_pool_members(self, pool_id):
    """
        Get the members of a pool

        :param pool: The instance of a pool
        :type  pool: ``NttCisPool``

        :returns: Returns an ``list`` of ``NttCisPoolMember``
        :rtype: ``list`` of ``NttCisPoolMember``
        """
    members = self.connection.request_with_orgId_api_2('networkDomainVip/poolMember?poolId=%s' % pool_id).object
    return self._to_members(members)