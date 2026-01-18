from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
def ex_list_listeners(self, balancer):
    """
        Get all listener related to the given balancer

        :param balancer: the balancer to list listeners
        :type balancer: ``LoadBalancer``

        :return: a list of listeners
        :rtype: ``list`` of ``SLBLoadBalancerListener``
        """
    attribute = self.ex_get_balancer_attribute(balancer)
    listeners = [SLBLoadBalancerListener(each['ListenerPort'], None, None, None) for each in attribute.listeners]
    return listeners