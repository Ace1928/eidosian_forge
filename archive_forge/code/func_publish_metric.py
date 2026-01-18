import abc
import time
import warnings
from collections import namedtuple
from functools import wraps
from typing import Dict, Optional
def publish_metric(metric_group: str, metric_name: str, metric_value: int):
    warnings.warn('Deprecated, use put_metric(metric_group)(metric_name, metric_value) instead')
    metric_stream = getStream(metric_group)
    metric_stream.add_value(metric_name, metric_value)