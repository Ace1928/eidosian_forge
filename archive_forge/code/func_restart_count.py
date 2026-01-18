from pprint import pformat
from six import iteritems
import re
@restart_count.setter
def restart_count(self, restart_count):
    """
        Sets the restart_count of this V1ContainerStatus.
        The number of times the container has been restarted, currently based on
        the number of dead containers that have not yet been removed. Note that
        this is calculated from dead containers. But those containers are
        subject to garbage collection. This value will get capped at 5 by GC.

        :param restart_count: The restart_count of this V1ContainerStatus.
        :type: int
        """
    if restart_count is None:
        raise ValueError('Invalid value for `restart_count`, must not be `None`')
    self._restart_count = restart_count