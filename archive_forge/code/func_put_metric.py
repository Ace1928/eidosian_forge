import abc
import time
import warnings
from collections import namedtuple
from functools import wraps
from typing import Dict, Optional
def put_metric(metric_name: str, metric_value: int, metric_group: str='torchelastic'):
    """
    Publish a metric data point.

    Usage

    ::

     put_metric("metric_name", 1)
     put_metric("metric_name", 1, "metric_group_name")
    """
    getStream(metric_group).add_value(metric_name, metric_value)