import abc
import time
import warnings
from collections import namedtuple
from functools import wraps
from typing import Dict, Optional
def getStream(group: str):
    if group in _metrics_map:
        handler = _metrics_map[group]
    else:
        handler = _default_metrics_handler
    return MetricStream(group, handler)