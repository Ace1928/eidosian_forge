from pprint import pformat
from six import iteritems
import re
@pods.setter
def pods(self, pods):
    """
        Sets the pods of this V2beta1MetricSpec.
        pods refers to a metric describing each pod in the current scale target
        (for example, transactions-processed-per-second).  The values will be
        averaged together before being compared to the target value.

        :param pods: The pods of this V2beta1MetricSpec.
        :type: V2beta1PodsMetricSource
        """
    self._pods = pods