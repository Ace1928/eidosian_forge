from pprint import pformat
from six import iteritems
import re
@allowed_host_paths.setter
def allowed_host_paths(self, allowed_host_paths):
    """
        Sets the allowed_host_paths of this PolicyV1beta1PodSecurityPolicySpec.
        allowedHostPaths is a white list of allowed host paths. Empty indicates
        that all host paths may be used.

        :param allowed_host_paths: The allowed_host_paths of this
        PolicyV1beta1PodSecurityPolicySpec.
        :type: list[PolicyV1beta1AllowedHostPath]
        """
    self._allowed_host_paths = allowed_host_paths