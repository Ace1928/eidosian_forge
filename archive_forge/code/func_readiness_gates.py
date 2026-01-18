from pprint import pformat
from six import iteritems
import re
@readiness_gates.setter
def readiness_gates(self, readiness_gates):
    """
        Sets the readiness_gates of this V1PodSpec.
        If specified, all readiness gates will be evaluated for pod readiness. A
        pod is ready when all its containers are ready AND all conditions
        specified in the readiness gates have status equal to "True" More
        info:
        https://git.k8s.io/enhancements/keps/sig-network/0007-pod-ready%2B%2B.md

        :param readiness_gates: The readiness_gates of this V1PodSpec.
        :type: list[V1PodReadinessGate]
        """
    self._readiness_gates = readiness_gates