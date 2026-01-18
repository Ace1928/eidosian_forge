from pprint import pformat
from six import iteritems
import re
@divisor.setter
def divisor(self, divisor):
    """
        Sets the divisor of this V1ResourceFieldSelector.
        Specifies the output format of the exposed resources, defaults to "1"

        :param divisor: The divisor of this V1ResourceFieldSelector.
        :type: str
        """
    self._divisor = divisor