from typing import TYPE_CHECKING, Callable, Union
import pandas
import ray
from modin.config import LazyExecution
from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from modin.core.execution.ray.common import MaterializationHook, RayWrapper
from modin.core.execution.ray.common.deferred_execution import (
from modin.core.execution.ray.common.utils import ObjectIDType
from modin.logging import disable_logging, get_logger
from modin.pandas.indexing import compute_sliced_len
from modin.utils import _inherit_docstrings
def eager_exec(self, func, *args, length=None, width=None, **kwargs):
    return self._eager_exec_func(func, *args, **kwargs)