from pprint import pformat
from six import iteritems
import re
@host_network.setter
def host_network(self, host_network):
    """
        Sets the host_network of this PolicyV1beta1PodSecurityPolicySpec.
        hostNetwork determines if the policy allows the use of HostNetwork in
        the pod spec.

        :param host_network: The host_network of this
        PolicyV1beta1PodSecurityPolicySpec.
        :type: bool
        """
    self._host_network = host_network