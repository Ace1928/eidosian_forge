import logging
import sys
import time
import threading
from .kafka_metric import KafkaMetric
from .measurable import AnonMeasurable
from .metric_config import MetricConfig
from .metric_name import MetricName
from .stats import Sensor
def add_reporter(self, reporter):
    """Add a MetricReporter"""
    with self._lock:
        reporter.init(list(self.metrics.values()))
        self._reporters.append(reporter)