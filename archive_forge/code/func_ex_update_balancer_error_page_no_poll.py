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
def ex_update_balancer_error_page_no_poll(self, balancer, page_content):
    """
        Updates a Balancer's custom error page.  This method returns
        immediately.

        :param balancer: Balancer to update the custom error page for.
        :type  balancer: :class:`LoadBalancer`

        :param page_content: HTML content for the custom error page.
        :type  page_content: ``str``

        :return: Returns whether the update request was accepted.
        :rtype: ``bool``
        """
    uri = '/loadbalancers/%s/errorpage' % balancer.id
    resp = self.connection.request(uri, method='PUT', data=json.dumps({'errorpage': {'content': page_content}}))
    return resp.status == httplib.ACCEPTED