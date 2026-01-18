from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
import ray
from .ref_bundle import RefBundle
from ray._raylet import ObjectRefGenerator
from ray.data._internal.execution.interfaces.execution_options import (
from ray.data._internal.execution.interfaces.op_runtime_metrics import OpRuntimeMetrics
from ray.data._internal.logical.interfaces import Operator
from ray.data._internal.stats import StatsDict
from ray.data.context import DataContext
def task_index(self) -> int:
    """Return the index of the task."""
    return self._task_index