from pprint import pformat
from six import iteritems
import re
@current_replicas.setter
def current_replicas(self, current_replicas):
    """
        Sets the current_replicas of this V1beta1StatefulSetStatus.
        currentReplicas is the number of Pods created by the StatefulSet
        controller from the StatefulSet version indicated by currentRevision.

        :param current_replicas: The current_replicas of this
        V1beta1StatefulSetStatus.
        :type: int
        """
    self._current_replicas = current_replicas