import logging
import sys
import time
import threading
from .kafka_metric import KafkaMetric
from .measurable import AnonMeasurable
from .metric_config import MetricConfig
from .metric_name import MetricName
from .stats import Sensor
def get_sensor(self, name):
    """
        Get the sensor with the given name if it exists

        Arguments:
            name (str): The name of the sensor

        Returns:
            Sensor: The sensor or None if no such sensor exists
        """
    if not name:
        raise ValueError('name must be non-empty')
    return self._sensors.get(name, None)