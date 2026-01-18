from pprint import pformat
from six import iteritems
import re
@taints.setter
def taints(self, taints):
    """
        Sets the taints of this V1NodeSpec.
        If specified, the node's taints.

        :param taints: The taints of this V1NodeSpec.
        :type: list[V1Taint]
        """
    self._taints = taints