from pprint import pformat
from six import iteritems
import re
@termination_message_path.setter
def termination_message_path(self, termination_message_path):
    """
        Sets the termination_message_path of this V1Container.
        Optional: Path at which the file to which the container's termination
        message will be written is mounted into the container's filesystem.
        Message written is intended to be brief final status, such as an
        assertion failure message. Will be truncated by the node if greater than
        4096 bytes. The total message length across all containers will be
        limited to 12kb. Defaults to /dev/termination-log. Cannot be updated.

        :param termination_message_path: The termination_message_path of this
        V1Container.
        :type: str
        """
    self._termination_message_path = termination_message_path