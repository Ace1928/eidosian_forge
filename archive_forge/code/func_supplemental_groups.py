from pprint import pformat
from six import iteritems
import re
@supplemental_groups.setter
def supplemental_groups(self, supplemental_groups):
    """
        Sets the supplemental_groups of this V1PodSecurityContext.
        A list of groups applied to the first process run in each container, in
        addition to the container's primary GID.  If unspecified, no groups will
        be added to any container.

        :param supplemental_groups: The supplemental_groups of this
        V1PodSecurityContext.
        :type: list[int]
        """
    self._supplemental_groups = supplemental_groups