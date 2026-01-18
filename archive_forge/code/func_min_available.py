from pprint import pformat
from six import iteritems
import re
@min_available.setter
def min_available(self, min_available):
    """
        Sets the min_available of this V1beta1PodDisruptionBudgetSpec.
        An eviction is allowed if at least "minAvailable" pods selected by
        "selector" will still be available after the eviction, i.e. even in
        the absence of the evicted pod.  So for example you can prevent all
        voluntary evictions by specifying "100%".

        :param min_available: The min_available of this
        V1beta1PodDisruptionBudgetSpec.
        :type: object
        """
    self._min_available = min_available