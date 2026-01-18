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
def ex_update_balancer_connection_throttle_no_poll(self, balancer, connection_throttle):
    """
        Sets a Balancer's connection throttle.  This method returns
        immediately.

        :param balancer: Balancer to update connection throttle on.
        :type  balancer: :class:`LoadBalancer`

        :param connection_throttle: Connection Throttle for the balancer.
        :type  connection_throttle: :class:`RackspaceConnectionThrottle`

        :return: Returns whether the update request was accepted.
        :rtype: ``bool``
        """
    uri = '/loadbalancers/%s/connectionthrottle' % balancer.id
    resp = self.connection.request(uri, method='PUT', data=json.dumps(connection_throttle._to_dict()))
    return resp.status == httplib.ACCEPTED