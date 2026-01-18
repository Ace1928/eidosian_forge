import collections
import dataclasses
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from wandb.sdk.lib import telemetry
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
@staticmethod
def flatten_stats(sample: _Stats) -> dict:
    """Flatten _Stats object into a flat dict of numbers."""
    flattened = {}

    def helper(key: str, value: Any) -> None:
        if isinstance(value, (int, float)):
            ret = {f'{key}': value}
            flattened.update(ret)
            return
        elif isinstance(value, dict):
            for kk, vv in value.items():
                if isinstance(kk, int):
                    helper(f'{kk}.{key}', vv)
                else:
                    helper(f'{key}.{kk}', vv)
            return
        elif isinstance(value, list):
            for i, val in enumerate(value):
                helper(f'{i}.{key}', val)
    for kkk, vvv in dataclasses.asdict(sample).items():
        helper(kkk, vvv)
    return flattened