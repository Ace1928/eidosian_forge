from libcloud.utils.py3 import httplib
from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def ex_list_balancer_policies(self, balancer):
    """
        Return a list of policy description string.

        :rtype: ``list`` of ``str``
        """
    params = {'Action': 'DescribeLoadBalancerPolicies', 'LoadBalancerName': balancer.id}
    data = self.connection.request(ROOT, params=params).object
    return self._to_policies(data)