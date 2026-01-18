from pprint import pformat
from six import iteritems
import re
@pod_cidr.setter
def pod_cidr(self, pod_cidr):
    """
        Sets the pod_cidr of this V1NodeSpec.
        PodCIDR represents the pod IP range assigned to the node.

        :param pod_cidr: The pod_cidr of this V1NodeSpec.
        :type: str
        """
    self._pod_cidr = pod_cidr