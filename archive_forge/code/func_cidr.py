from pprint import pformat
from six import iteritems
import re
@cidr.setter
def cidr(self, cidr):
    """
        Sets the cidr of this V1beta1IPBlock.
        CIDR is a string representing the IP Block Valid examples are
        "192.168.1.1/24"

        :param cidr: The cidr of this V1beta1IPBlock.
        :type: str
        """
    if cidr is None:
        raise ValueError('Invalid value for `cidr`, must not be `None`')
    self._cidr = cidr