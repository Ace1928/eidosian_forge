from typing import List
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.base_physical_operator import NAryOperator
from ray.data._internal.stats import StatsDict
def get_stats(self) -> StatsDict:
    return self._stats