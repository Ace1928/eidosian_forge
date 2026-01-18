from pprint import pformat
from six import iteritems
import re
@disrupted_pods.setter
def disrupted_pods(self, disrupted_pods):
    """
        Sets the disrupted_pods of this V1beta1PodDisruptionBudgetStatus.
        DisruptedPods contains information about pods whose eviction was
        processed by the API server eviction subresource handler but has not yet
        been observed by the PodDisruptionBudget controller. A pod will be in
        this map from the time when the API server processed the eviction
        request to the time when the pod is seen by PDB controller as having
        been marked for deletion (or after a timeout). The key in the map is the
        name of the pod and the value is the time when the API server processed
        the eviction request. If the deletion didn't occur and a pod is still
        there it will be removed from the list automatically by
        PodDisruptionBudget controller after some time. If everything goes
        smooth this map should be empty for the most of the time. Large number
        of entries in the map may indicate problems with pod deletions.

        :param disrupted_pods: The disrupted_pods of this
        V1beta1PodDisruptionBudgetStatus.
        :type: dict(str, datetime)
        """
    self._disrupted_pods = disrupted_pods