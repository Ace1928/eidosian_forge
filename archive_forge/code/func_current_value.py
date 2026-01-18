from pprint import pformat
from six import iteritems
import re
@current_value.setter
def current_value(self, current_value):
    """
        Sets the current_value of this V2beta1ObjectMetricStatus.
        currentValue is the current value of the metric (as a quantity).

        :param current_value: The current_value of this
        V2beta1ObjectMetricStatus.
        :type: str
        """
    if current_value is None:
        raise ValueError('Invalid value for `current_value`, must not be `None`')
    self._current_value = current_value