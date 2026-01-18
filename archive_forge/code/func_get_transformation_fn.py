from typing import List, Optional
from ray.data._internal.execution.interfaces import (
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.stats import StatsDict
def get_transformation_fn(self) -> AllToAllTransformFn:
    return self._bulk_fn