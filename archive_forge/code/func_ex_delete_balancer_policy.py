from libcloud.utils.py3 import httplib
from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def ex_delete_balancer_policy(self, name, policy_name):
    """
        Delete a load balancer policy

        :param name: balancer name for which policy will be deleted
        :type  name: ``str``

        :param policy_name: The Mnemonic name for the policy being deleted
        :type  policy_name: ``str``
        """
    params = {'Action': 'DeleteLoadBalancerPolicy', 'LoadBalancerName': name, 'PolicyName': policy_name}
    response = self.connection.request(ROOT, params=params)
    return response.status == httplib.OK