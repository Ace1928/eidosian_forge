from pprint import pformat
from six import iteritems
import re
@operating_system.setter
def operating_system(self, operating_system):
    """
        Sets the operating_system of this V1NodeSystemInfo.
        The Operating System reported by the node

        :param operating_system: The operating_system of this V1NodeSystemInfo.
        :type: str
        """
    if operating_system is None:
        raise ValueError('Invalid value for `operating_system`, must not be `None`')
    self._operating_system = operating_system