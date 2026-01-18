from pprint import pformat
from six import iteritems
import re
@last_state.setter
def last_state(self, last_state):
    """
        Sets the last_state of this V1ContainerStatus.
        Details about the container's last termination condition.

        :param last_state: The last_state of this V1ContainerStatus.
        :type: V1ContainerState
        """
    self._last_state = last_state