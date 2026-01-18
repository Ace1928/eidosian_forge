from pprint import pformat
from six import iteritems
import re
@observed_generation.setter
def observed_generation(self, observed_generation):
    """
        Sets the observed_generation of this V1ReplicaSetStatus.
        ObservedGeneration reflects the generation of the most recently observed
        ReplicaSet.

        :param observed_generation: The observed_generation of this
        V1ReplicaSetStatus.
        :type: int
        """
    self._observed_generation = observed_generation