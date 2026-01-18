from pprint import pformat
from six import iteritems
import re
@host_pid.setter
def host_pid(self, host_pid):
    """
        Sets the host_pid of this PolicyV1beta1PodSecurityPolicySpec.
        hostPID determines if the policy allows the use of HostPID in the pod
        spec.

        :param host_pid: The host_pid of this
        PolicyV1beta1PodSecurityPolicySpec.
        :type: bool
        """
    self._host_pid = host_pid