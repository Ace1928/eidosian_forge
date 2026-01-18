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
def ex_update_balancer_health_monitor_no_poll(self, balancer, health_monitor):
    """
        Sets a Balancer's health monitor.  This method returns immediately.

        :param balancer: Balancer to update health monitor on.
        :type  balancer: :class:`LoadBalancer`

        :param health_monitor: Health Monitor for the balancer.
        :type  health_monitor: :class:`RackspaceHealthMonitor`

        :return: Returns whether the update request was accepted.
        :rtype: ``bool``
        """
    uri = '/loadbalancers/%s/healthmonitor' % balancer.id
    resp = self.connection.request(uri, method='PUT', data=json.dumps(health_monitor._to_dict()))
    return resp.status == httplib.ACCEPTED