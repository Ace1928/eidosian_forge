from pprint import pformat
from six import iteritems
import re
@default_add_capabilities.setter
def default_add_capabilities(self, default_add_capabilities):
    """
        Sets the default_add_capabilities of this
        PolicyV1beta1PodSecurityPolicySpec.
        defaultAddCapabilities is the default set of capabilities that will be
        added to the container unless the pod spec specifically drops the
        capability.  You may not list a capability in both
        defaultAddCapabilities and requiredDropCapabilities. Capabilities added
        here are implicitly allowed, and need not be included in the
        allowedCapabilities list.

        :param default_add_capabilities: The default_add_capabilities of this
        PolicyV1beta1PodSecurityPolicySpec.
        :type: list[str]
        """
    self._default_add_capabilities = default_add_capabilities