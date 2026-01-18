import logging
import sys
import time
import threading
from .kafka_metric import KafkaMetric
from .measurable import AnonMeasurable
from .metric_config import MetricConfig
from .metric_name import MetricName
from .stats import Sensor
def remove_metric(self, metric_name):
    """
        Remove a metric if it exists and return it. Return None otherwise.
        If a metric is removed, `metric_removal` will be invoked
        for each reporter.

        Arguments:
            metric_name (MetricName): The name of the metric

        Returns:
            KafkaMetric: the removed `KafkaMetric` or None if no such
                metric exists
        """
    with self._lock:
        metric = self._metrics.pop(metric_name, None)
        if metric:
            for reporter in self._reporters:
                reporter.metric_removal(metric)
        return metric