from pprint import pformat
from six import iteritems
import re
@run_as_user.setter
def run_as_user(self, run_as_user):
    """
        Sets the run_as_user of this V1SecurityContext.
        The UID to run the entrypoint of the container process. Defaults to user
        specified in image metadata if unspecified. May also be set in
        PodSecurityContext.  If set in both SecurityContext and
        PodSecurityContext, the value specified in SecurityContext takes
        precedence.

        :param run_as_user: The run_as_user of this V1SecurityContext.
        :type: int
        """
    self._run_as_user = run_as_user