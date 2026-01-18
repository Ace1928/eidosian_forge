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
def ex_update_balancer_no_poll(self, balancer, **kwargs):
    """
        Update balancer no poll.

        @inherits: :class:`Driver.update_balancer`
        """
    attrs = self._kwargs_to_mutable_attrs(**kwargs)
    resp = self.connection.request(action='/loadbalancers/%s' % balancer.id, method='PUT', data=json.dumps(attrs))
    return resp.status == httplib.ACCEPTED