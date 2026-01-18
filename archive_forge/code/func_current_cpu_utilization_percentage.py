from pprint import pformat
from six import iteritems
import re
@current_cpu_utilization_percentage.setter
def current_cpu_utilization_percentage(self, current_cpu_utilization_percentage):
    """
        Sets the current_cpu_utilization_percentage of this
        V1HorizontalPodAutoscalerStatus.
        current average CPU utilization over all pods, represented as a
        percentage of requested CPU, e.g. 70 means that an average pod is using
        now 70% of its requested CPU.

        :param current_cpu_utilization_percentage: The
        current_cpu_utilization_percentage of this
        V1HorizontalPodAutoscalerStatus.
        :type: int
        """
    self._current_cpu_utilization_percentage = current_cpu_utilization_percentage