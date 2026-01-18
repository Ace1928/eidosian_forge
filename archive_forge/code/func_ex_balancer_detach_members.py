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
def ex_balancer_detach_members(self, balancer, members):
    """
        Detaches a list of members from a balancer (the API supports up to 10).
        This method blocks until the detach request has been processed and the
        balancer is in a RUNNING state again.

        :param balancer: The Balancer to detach members from.
        :type  balancer: :class:`LoadBalancer`

        :param members: A list of Members to detach.
        :type  members: ``list`` of :class:`Member`

        :return: Updated Balancer.
        :rtype: :class:`LoadBalancer`
        """
    accepted = self.ex_balancer_detach_members_no_poll(balancer, members)
    if not accepted:
        msg = 'Detach members request was not accepted'
        raise LibcloudError(msg, driver=self)
    return self._get_updated_balancer(balancer)