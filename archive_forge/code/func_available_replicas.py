from pprint import pformat
from six import iteritems
import re
@available_replicas.setter
def available_replicas(self, available_replicas):
    """
        Sets the available_replicas of this V1ReplicaSetStatus.
        The number of available replicas (ready for at least minReadySeconds)
        for this replica set.

        :param available_replicas: The available_replicas of this
        V1ReplicaSetStatus.
        :type: int
        """
    self._available_replicas = available_replicas