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
def ex_destroy_balancer_access_rule(self, balancer, rule):
    """
        Removes an access rule from a Balancer's access list.  This method
        blocks until the update request has been processed and the balancer
        is in a RUNNING state again.

        :param balancer: Balancer to remove the access rule from.
        :type  balancer: :class:`LoadBalancer`

        :param rule: Access Rule to remove from the balancer.
        :type  rule: :class:`RackspaceAccessRule`

        :return: Updated Balancer.
        :rtype: :class:`LoadBalancer`
        """
    accepted = self.ex_destroy_balancer_access_rule_no_poll(balancer, rule)
    if not accepted:
        msg = 'Delete access rule not accepted'
        raise LibcloudError(msg, driver=self)
    return self._get_updated_balancer(balancer)