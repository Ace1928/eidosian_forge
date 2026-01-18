from pprint import pformat
from six import iteritems
import re
@average_value.setter
def average_value(self, average_value):
    """
        Sets the average_value of this V2beta1ObjectMetricStatus.
        averageValue is the current value of the average of the metric across
        all relevant pods (as a quantity)

        :param average_value: The average_value of this
        V2beta1ObjectMetricStatus.
        :type: str
        """
    self._average_value = average_value