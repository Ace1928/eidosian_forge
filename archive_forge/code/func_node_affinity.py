from pprint import pformat
from six import iteritems
import re
@node_affinity.setter
def node_affinity(self, node_affinity):
    """
        Sets the node_affinity of this V1Affinity.
        Describes node affinity scheduling rules for the pod.

        :param node_affinity: The node_affinity of this V1Affinity.
        :type: V1NodeAffinity
        """
    self._node_affinity = node_affinity