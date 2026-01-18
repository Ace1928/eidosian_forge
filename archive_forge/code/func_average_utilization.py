from pprint import pformat
from six import iteritems
import re
@average_utilization.setter
def average_utilization(self, average_utilization):
    """
        Sets the average_utilization of this V2beta2MetricValueStatus.
        currentAverageUtilization is the current value of the average of the
        resource metric across all relevant pods, represented as a percentage of
        the requested value of the resource for the pods.

        :param average_utilization: The average_utilization of this
        V2beta2MetricValueStatus.
        :type: int
        """
    self._average_utilization = average_utilization