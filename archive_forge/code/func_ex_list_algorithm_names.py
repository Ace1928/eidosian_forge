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
def ex_list_algorithm_names(self):
    """
        Lists algorithms supported by the API.  Returned as strings because
        this list may change in the future.

        :rtype: ``list`` of ``str``
        """
    response = self.connection.request('/loadbalancers/algorithms')
    return [a['name'].upper() for a in response.object['algorithms']]