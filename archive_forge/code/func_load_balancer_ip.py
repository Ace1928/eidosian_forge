from pprint import pformat
from six import iteritems
import re
@load_balancer_ip.setter
def load_balancer_ip(self, load_balancer_ip):
    """
        Sets the load_balancer_ip of this V1ServiceSpec.
        Only applies to Service Type: LoadBalancer LoadBalancer will get created
        with the IP specified in this field. This feature depends on whether the
        underlying cloud-provider supports specifying the loadBalancerIP when a
        load balancer is created. This field will be ignored if the
        cloud-provider does not support the feature.

        :param load_balancer_ip: The load_balancer_ip of this V1ServiceSpec.
        :type: str
        """
    self._load_balancer_ip = load_balancer_ip