from pprint import pformat
from six import iteritems
import re
@allowed_capabilities.setter
def allowed_capabilities(self, allowed_capabilities):
    """
        Sets the allowed_capabilities of this
        PolicyV1beta1PodSecurityPolicySpec.
        allowedCapabilities is a list of capabilities that can be requested to
        add to the container. Capabilities in this field may be added at the pod
        author's discretion. You must not list a capability in both
        allowedCapabilities and requiredDropCapabilities.

        :param allowed_capabilities: The allowed_capabilities of this
        PolicyV1beta1PodSecurityPolicySpec.
        :type: list[str]
        """
    self._allowed_capabilities = allowed_capabilities