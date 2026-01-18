import json
import logging
import shutil
import subprocess
import sys
import threading
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Union
from wandb.sdk.lib import telemetry
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
@staticmethod
def parse_stats(stats: Dict[str, str]) -> _Stats:
    """Parse stats from rocm-smi output."""
    parsed_stats: _Stats = {}
    try:
        parsed_stats['gpu'] = float(stats.get('GPU use (%)'))
    except (TypeError, ValueError):
        logger.warning('Could not parse GPU usage as float')
    try:
        parsed_stats['memoryAllocated'] = float(stats.get('GPU memory use (%)'))
    except (TypeError, ValueError):
        logger.warning('Could not parse GPU memory allocation as float')
    try:
        parsed_stats['temp'] = float(stats.get('Temperature (Sensor memory) (C)'))
    except (TypeError, ValueError):
        logger.warning('Could not parse GPU temperature as float')
    try:
        parsed_stats['powerWatts'] = float(stats.get('Average Graphics Package Power (W)'))
    except (TypeError, ValueError):
        logger.warning('Could not parse GPU power as float')
    try:
        parsed_stats['powerPercent'] = float(stats.get('Average Graphics Package Power (W)')) / float(stats.get('Max Graphics Package Power (W)')) * 100
    except (TypeError, ValueError):
        logger.warning('Could not parse GPU average/max power as float')
    return parsed_stats