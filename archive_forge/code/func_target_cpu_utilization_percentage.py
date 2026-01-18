from pprint import pformat
from six import iteritems
import re
@target_cpu_utilization_percentage.setter
def target_cpu_utilization_percentage(self, target_cpu_utilization_percentage):
    """
        Sets the target_cpu_utilization_percentage of this
        V1HorizontalPodAutoscalerSpec.
        target average CPU utilization (represented as a percentage of requested
        CPU) over all the pods; if not specified the default autoscaling policy
        will be used.

        :param target_cpu_utilization_percentage: The
        target_cpu_utilization_percentage of this V1HorizontalPodAutoscalerSpec.
        :type: int
        """
    self._target_cpu_utilization_percentage = target_cpu_utilization_percentage