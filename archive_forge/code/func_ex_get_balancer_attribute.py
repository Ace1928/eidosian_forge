from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
def ex_get_balancer_attribute(self, balancer):
    """
        Get balancer attribute

        :param balancer: the balancer to get attribute
        :type balancer: ``LoadBalancer``

        :return: the balancer attribute
        :rtype: ``SLBLoadBalancerAttribute``
        """
    params = {'Action': 'DescribeLoadBalancerAttribute', 'LoadBalancerId': balancer.id}
    resp_body = self.connection.request(self.path, params).object
    attribute = self._to_balancer_attribute(resp_body)
    return attribute