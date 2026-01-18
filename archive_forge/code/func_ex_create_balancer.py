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
def ex_create_balancer(self, name, members, protocol='http', port=80, algorithm=DEFAULT_ALGORITHM, vip='PUBLIC'):
    """
        Creates a new load balancer instance

        :param name: Name of the new load balancer (required)
        :type  name: ``str``

        :param members: ``list`` of:class:`Member`s to attach to balancer
        :type  members: ``list`` of :class:`Member`

        :param protocol: Loadbalancer protocol, defaults to http.
        :type  protocol: ``str``

        :param port: Port the load balancer should listen on, defaults to 80
        :type  port: ``str``

        :param algorithm: Load balancing algorithm, defaults to
                            LBAlgorithm.ROUND_ROBIN
        :type  algorithm: :class:`Algorithm`

        :param vip: Virtual ip type of PUBLIC, SERVICENET, or ID of a virtual
                      ip
        :type  vip: ``str``

        :rtype: :class:`LoadBalancer`
        """
    balancer_attrs = self._kwargs_to_mutable_attrs(name=name, protocol=protocol, port=port, algorithm=algorithm, vip=vip)
    balancer_attrs.update({'nodes': [self._member_attributes(member) for member in members]})
    balancer_object = {'loadBalancer': balancer_attrs}
    resp = self.connection.request('/loadbalancers', method='POST', data=json.dumps(balancer_object))
    return self._to_balancer(resp.object['loadBalancer'])