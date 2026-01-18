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
def ex_balancer_attach_members(self, balancer, members):
    """
        Attaches a list of members to a load balancer.

        :param balancer: The Balancer to which members will be attached.
        :type  balancer: :class:`LoadBalancer`

        :param members: A list of Members to attach.
        :type  members: ``list`` of :class:`Member`

        :rtype: ``list`` of :class:`Member`
        """
    member_objects = {'nodes': [self._member_attributes(member) for member in members]}
    uri = '/loadbalancers/%s/nodes' % balancer.id
    resp = self.connection.request(uri, method='POST', data=json.dumps(member_objects))
    return self._to_members(resp.object, balancer)