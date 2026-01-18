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
def ex_create_balancer_access_rule(self, balancer, rule):
    """
        Adds an access rule to a Balancer's access list.  This method blocks
        until the update request has been processed and the balancer is in a
        RUNNING state again.

        :param balancer: Balancer to create the access rule for.
        :type balancer: :class:`LoadBalancer`

        :param rule: Access Rule to add to the balancer.
        :type rule: :class:`RackspaceAccessRule`

        :return: The created access rule.
        :rtype: :class:`RackspaceAccessRule`
        """
    accepted = self.ex_create_balancer_access_rule_no_poll(balancer, rule)
    if not accepted:
        msg = 'Create access rule not accepted'
        raise LibcloudError(msg, driver=self)
    balancer = self._get_updated_balancer(balancer)
    access_list = balancer.extra['accessList']
    created_rule = self._find_matching_rule(rule, access_list)
    if not created_rule:
        raise LibcloudError('Could not find created rule')
    return created_rule