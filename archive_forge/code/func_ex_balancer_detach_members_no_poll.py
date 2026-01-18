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
def ex_balancer_detach_members_no_poll(self, balancer, members):
    """
        Detaches a list of members from a balancer (the API supports up to 10).
        This method returns immediately.

        :param balancer: The Balancer to detach members from.
        :type  balancer: :class:`LoadBalancer`

        :param members: A list of Members to detach.
        :type  members: ``list`` of :class:`Member`

        :return: Returns whether the detach request was accepted.
        :rtype: ``bool``
        """
    uri = '/loadbalancers/%s/nodes' % balancer.id
    ids = [('id', member.id) for member in members]
    resp = self.connection.request(uri, method='DELETE', params=ids)
    return resp.status == httplib.ACCEPTED