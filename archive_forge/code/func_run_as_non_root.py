from pprint import pformat
from six import iteritems
import re
@run_as_non_root.setter
def run_as_non_root(self, run_as_non_root):
    """
        Sets the run_as_non_root of this V1SecurityContext.
        Indicates that the container must run as a non-root user. If true, the
        Kubelet will validate the image at runtime to ensure that it does not
        run as UID 0 (root) and fail to start the container if it does. If unset
        or false, no such validation will be performed. May also be set in
        PodSecurityContext.  If set in both SecurityContext and
        PodSecurityContext, the value specified in SecurityContext takes
        precedence.

        :param run_as_non_root: The run_as_non_root of this V1SecurityContext.
        :type: bool
        """
    self._run_as_non_root = run_as_non_root