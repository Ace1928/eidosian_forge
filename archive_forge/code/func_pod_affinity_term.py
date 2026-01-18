from pprint import pformat
from six import iteritems
import re
@pod_affinity_term.setter
def pod_affinity_term(self, pod_affinity_term):
    """
        Sets the pod_affinity_term of this V1WeightedPodAffinityTerm.
        Required. A pod affinity term, associated with the corresponding weight.

        :param pod_affinity_term: The pod_affinity_term of this
        V1WeightedPodAffinityTerm.
        :type: V1PodAffinityTerm
        """
    if pod_affinity_term is None:
        raise ValueError('Invalid value for `pod_affinity_term`, must not be `None`')
    self._pod_affinity_term = pod_affinity_term