from pprint import pformat
from six import iteritems
import re
@se_linux_options.setter
def se_linux_options(self, se_linux_options):
    """
        Sets the se_linux_options of this V1SecurityContext.
        The SELinux context to be applied to the container. If unspecified, the
        container runtime will allocate a random SELinux context for each
        container.  May also be set in PodSecurityContext.  If set in both
        SecurityContext and PodSecurityContext, the value specified in
        SecurityContext takes precedence.

        :param se_linux_options: The se_linux_options of this V1SecurityContext.
        :type: V1SELinuxOptions
        """
    self._se_linux_options = se_linux_options