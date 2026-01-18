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
def ex_list_protocols_with_default_ports(self):
    """
        List protocols with default ports.

        :rtype: ``list`` of ``tuple``
        :return: A list of protocols with default ports included.
        """
    return self._to_protocols_with_default_ports(self.connection.request('/loadbalancers/protocols').object)