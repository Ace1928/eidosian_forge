from pprint import pformat
from six import iteritems
import re
@min_replicas.setter
def min_replicas(self, min_replicas):
    """
        Sets the min_replicas of this V2beta2HorizontalPodAutoscalerSpec.
        minReplicas is the lower limit for the number of replicas to which the
        autoscaler can scale down. It defaults to 1 pod.

        :param min_replicas: The min_replicas of this
        V2beta2HorizontalPodAutoscalerSpec.
        :type: int
        """
    self._min_replicas = min_replicas