from libcloud.utils.py3 import httplib
from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def ex_set_balancer_policies_listener(self, name, port, policies):
    """
        Associates, updates, or disables a policy with a listener on
        the load balancer

        :param name: balancer name to set policies for listerner
        :type  name: ``str``

        :param port: port to use
        :type  port: ``str``

        :param policies: List of policies to be associated with the balancer
        :type  policies: ``string list``
        """
    params = {'Action': 'SetLoadBalancerPoliciesOfListener', 'LoadBalancerName': name, 'LoadBalancerPort': str(port)}
    if policies:
        params = self._create_list_params(params, policies, 'PolicyNames.member.%d')
    response = self.connection.request(ROOT, params=params)
    return response.status == httplib.OK