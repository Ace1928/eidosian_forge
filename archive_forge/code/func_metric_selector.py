from pprint import pformat
from six import iteritems
import re
@metric_selector.setter
def metric_selector(self, metric_selector):
    """
        Sets the metric_selector of this V2beta1ExternalMetricSource.
        metricSelector is used to identify a specific time series within a given
        metric.

        :param metric_selector: The metric_selector of this
        V2beta1ExternalMetricSource.
        :type: V1LabelSelector
        """
    self._metric_selector = metric_selector