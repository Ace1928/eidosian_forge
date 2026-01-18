from pprint import pformat
from six import iteritems
import re
@run_as_group.setter
def run_as_group(self, run_as_group):
    """
        Sets the run_as_group of this V1SecurityContext.
        The GID to run the entrypoint of the container process. Uses runtime
        default if unset. May also be set in PodSecurityContext.  If set in both
        SecurityContext and PodSecurityContext, the value specified in
        SecurityContext takes precedence.

        :param run_as_group: The run_as_group of this V1SecurityContext.
        :type: int
        """
    self._run_as_group = run_as_group