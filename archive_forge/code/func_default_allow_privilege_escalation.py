from pprint import pformat
from six import iteritems
import re
@default_allow_privilege_escalation.setter
def default_allow_privilege_escalation(self, default_allow_privilege_escalation):
    """
        Sets the default_allow_privilege_escalation of this
        PolicyV1beta1PodSecurityPolicySpec.
        defaultAllowPrivilegeEscalation controls the default setting for whether
        a process can gain more privileges than its parent process.

        :param default_allow_privilege_escalation: The
        default_allow_privilege_escalation of this
        PolicyV1beta1PodSecurityPolicySpec.
        :type: bool
        """
    self._default_allow_privilege_escalation = default_allow_privilege_escalation