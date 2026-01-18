from pprint import pformat
from six import iteritems
import re
@read_only_root_filesystem.setter
def read_only_root_filesystem(self, read_only_root_filesystem):
    """
        Sets the read_only_root_filesystem of this V1SecurityContext.
        Whether this container has a read-only root filesystem. Default is
        false.

        :param read_only_root_filesystem: The read_only_root_filesystem of this
        V1SecurityContext.
        :type: bool
        """
    self._read_only_root_filesystem = read_only_root_filesystem