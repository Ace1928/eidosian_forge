from pprint import pformat
from six import iteritems
import re
@allowed_proc_mount_types.setter
def allowed_proc_mount_types(self, allowed_proc_mount_types):
    """
        Sets the allowed_proc_mount_types of this
        PolicyV1beta1PodSecurityPolicySpec.
        AllowedProcMountTypes is a whitelist of allowed ProcMountTypes. Empty or
        nil indicates that only the DefaultProcMountType may be used. This
        requires the ProcMountType feature flag to be enabled.

        :param allowed_proc_mount_types: The allowed_proc_mount_types of this
        PolicyV1beta1PodSecurityPolicySpec.
        :type: list[str]
        """
    self._allowed_proc_mount_types = allowed_proc_mount_types