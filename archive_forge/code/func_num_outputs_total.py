from typing import List
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.base_physical_operator import NAryOperator
from ray.data._internal.stats import StatsDict
def num_outputs_total(self) -> int:
    num_outputs = 0
    for input_op in self.input_dependencies:
        num_outputs += input_op.num_outputs_total()
    return num_outputs