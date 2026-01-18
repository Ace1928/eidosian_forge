from pprint import pformat
from six import iteritems
import re
@hostnames.setter
def hostnames(self, hostnames):
    """
        Sets the hostnames of this V1HostAlias.
        Hostnames for the above IP address.

        :param hostnames: The hostnames of this V1HostAlias.
        :type: list[str]
        """
    self._hostnames = hostnames