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
def ex_update_balancer_health_monitor(self, balancer, health_monitor):
    """
        Sets a Balancer's health monitor.  This method blocks until the update
        request has been processed and the balancer is in a RUNNING state
        again.

        :param balancer: Balancer to update.
        :type  balancer: :class:`LoadBalancer`

        :param health_monitor: Health Monitor for the balancer.
        :type  health_monitor: :class:`RackspaceHealthMonitor`

        :return: Updated Balancer.
        :rtype: :class:`LoadBalancer`
        """
    accepted = self.ex_update_balancer_health_monitor_no_poll(balancer, health_monitor)
    if not accepted:
        msg = 'Update health monitor request not accepted'
        raise LibcloudError(msg, driver=self)
    return self._get_updated_balancer(balancer)