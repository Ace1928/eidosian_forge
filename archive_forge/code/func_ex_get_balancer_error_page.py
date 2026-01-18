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
def ex_get_balancer_error_page(self, balancer):
    """
        List error page configured for the specified load balancer.

        :param balancer: Balancer which should be used
        :type balancer: :class:`LoadBalancer`

        :rtype: ``str``
        """
    uri = '/loadbalancers/%s/errorpage' % balancer.id
    resp = self.connection.request(uri)
    return resp.object['errorpage']['content']