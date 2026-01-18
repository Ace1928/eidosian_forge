from pprint import pformat
from six import iteritems
import re
@metric_name.setter
def metric_name(self, metric_name):
    """
        Sets the metric_name of this V2beta1ObjectMetricStatus.
        metricName is the name of the metric in question.

        :param metric_name: The metric_name of this V2beta1ObjectMetricStatus.
        :type: str
        """
    if metric_name is None:
        raise ValueError('Invalid value for `metric_name`, must not be `None`')
    self._metric_name = metric_name