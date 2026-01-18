from datetime import datetime
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import reverse_dict
from libcloud.common.base import JsonResponse, PollingConnection
from libcloud.common.types import LibcloudError
from libcloud.common.openstack import OpenStackDriverMixin
from libcloud.common.rackspace import AUTH_URL
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, MemberCondition
from libcloud.compute.drivers.rackspace import RackspaceConnection
def ex_balancer_access_list(self, balancer):
    """
        List the access list.

        :param balancer: Balancer which should be used
        :type balancer: :class:`LoadBalancer`

        :rtype: ``list`` of :class:`RackspaceAccessRule`
        """
    uri = '/loadbalancers/%s/accesslist' % balancer.id
    resp = self.connection.request(uri)
    return [self._to_access_rule(el) for el in resp.object['accessList']]