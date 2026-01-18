from pprint import pformat
from six import iteritems
import re
@current_average_value.setter
def current_average_value(self, current_average_value):
    """
        Sets the current_average_value of this V2beta1ResourceMetricStatus.
        currentAverageValue is the current value of the average of the resource
        metric across all relevant pods, as a raw value (instead of as a
        percentage of the request), similar to the "pods" metric source type.
        It will always be set, regardless of the corresponding metric
        specification.

        :param current_average_value: The current_average_value of this
        V2beta1ResourceMetricStatus.
        :type: str
        """
    if current_average_value is None:
        raise ValueError('Invalid value for `current_average_value`, must not be `None`')
    self._current_average_value = current_average_value