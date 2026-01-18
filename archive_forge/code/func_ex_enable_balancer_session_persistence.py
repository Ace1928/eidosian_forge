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
def ex_enable_balancer_session_persistence(self, balancer):
    """
        Enables session persistence for a Balancer by setting the persistence
        type to 'HTTP_COOKIE'.  This method blocks until the enable request
        has been processed and the balancer is in a RUNNING state again.

        :param balancer: Balancer to enable session persistence on.
        :type  balancer: :class:`LoadBalancer`

        :return: Updated Balancer.
        :rtype: :class:`LoadBalancer`
        """
    if not self.ex_enable_balancer_session_persistence_no_poll(balancer):
        msg = 'Enable session persistence request not accepted'
        raise LibcloudError(msg, driver=self)
    return self._get_updated_balancer(balancer)