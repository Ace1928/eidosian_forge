import bisect
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, List, Optional
from ray.serve._private.constants import SERVE_LOGGER_NAME
def add_metrics_point(self, data_points: Dict[str, float], timestamp: float):
    """Push new data points to the store.

        Args:
            data_points: dictionary containing the metrics values. The
              key should be a string that uniquely identifies this time series
              and to be used to perform aggregation.
            timestamp: the unix epoch timestamp the metrics are
              collected at.
        """
    for name, value in data_points.items():
        bisect.insort(a=self.data[name], x=TimeStampedValue(timestamp, value))