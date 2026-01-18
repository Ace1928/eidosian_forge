from pprint import pformat
from six import iteritems
import re
@current_metrics.setter
def current_metrics(self, current_metrics):
    """
        Sets the current_metrics of this V2beta1HorizontalPodAutoscalerStatus.
        currentMetrics is the last read state of the metrics used by this
        autoscaler.

        :param current_metrics: The current_metrics of this
        V2beta1HorizontalPodAutoscalerStatus.
        :type: list[V2beta1MetricStatus]
        """
    self._current_metrics = current_metrics