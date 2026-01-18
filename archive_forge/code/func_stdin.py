from pprint import pformat
from six import iteritems
import re
@stdin.setter
def stdin(self, stdin):
    """
        Sets the stdin of this V1Container.
        Whether this container should allocate a buffer for stdin in the
        container runtime. If this is not set, reads from stdin in the container
        will always result in EOF. Default is false.

        :param stdin: The stdin of this V1Container.
        :type: bool
        """
    self._stdin = stdin