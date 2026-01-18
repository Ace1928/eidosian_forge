from pprint import pformat
from six import iteritems
import re
@pod_anti_affinity.setter
def pod_anti_affinity(self, pod_anti_affinity):
    """
        Sets the pod_anti_affinity of this V1Affinity.
        Describes pod anti-affinity scheduling rules (e.g. avoid putting this
        pod in the same node, zone, etc. as some other pod(s)).

        :param pod_anti_affinity: The pod_anti_affinity of this V1Affinity.
        :type: V1PodAntiAffinity
        """
    self._pod_anti_affinity = pod_anti_affinity