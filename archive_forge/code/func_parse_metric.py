import threading
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union
import wandb
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
@staticmethod
def parse_metric(key: str, value: str) -> Optional[Tuple[str, Union[int, float]]]:
    metric_suffixes = {'temp': 'C', 'clock': 'MHz', 'power': 'W', 'utilisation': '%', 'utilisation (session)': '%', 'speed': 'GT/s'}
    for metric, suffix in metric_suffixes.items():
        if key.endswith(metric) and value.endswith(suffix):
            value = value[:-len(suffix)]
            key = f'{key} ({suffix})'
    try:
        float_value = float(value)
        num_value = int(float_value) if float_value.is_integer() else float_value
    except ValueError:
        return None
    return (key, num_value)