from pprint import pformat
from six import iteritems
import re
@preferred_during_scheduling_ignored_during_execution.setter
def preferred_during_scheduling_ignored_during_execution(self, preferred_during_scheduling_ignored_during_execution):
    """
        Sets the preferred_during_scheduling_ignored_during_execution of this
        V1PodAffinity.
        The scheduler will prefer to schedule pods to nodes that satisfy the
        affinity expressions specified by this field, but it may choose a node
        that violates one or more of the expressions. The node that is most
        preferred is the one with the greatest sum of weights, i.e. for each
        node that meets all of the scheduling requirements (resource request,
        requiredDuringScheduling affinity expressions, etc.), compute a sum by
        iterating through the elements of this field and adding "weight" to
        the sum if the node has pods which matches the corresponding
        podAffinityTerm; the node(s) with the highest sum are the most
        preferred.

        :param preferred_during_scheduling_ignored_during_execution: The
        preferred_during_scheduling_ignored_during_execution of this
        V1PodAffinity.
        :type: list[V1WeightedPodAffinityTerm]
        """
    self._preferred_during_scheduling_ignored_during_execution = preferred_during_scheduling_ignored_during_execution