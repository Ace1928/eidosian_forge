from pprint import pformat
from six import iteritems
import re
@target_value.setter
def target_value(self, target_value):
    """
        Sets the target_value of this V2beta1ExternalMetricSource.
        targetValue is the target value of the metric (as a quantity). Mutually
        exclusive with TargetAverageValue.

        :param target_value: The target_value of this
        V2beta1ExternalMetricSource.
        :type: str
        """
    self._target_value = target_value