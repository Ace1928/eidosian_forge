from pprint import pformat
from six import iteritems
import re
@terminated.setter
def terminated(self, terminated):
    """
        Sets the terminated of this V1ContainerState.
        Details about a terminated container

        :param terminated: The terminated of this V1ContainerState.
        :type: V1ContainerStateTerminated
        """
    self._terminated = terminated