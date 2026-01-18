import abc
from typing import Optional
from ray.data._internal.logical.interfaces import LogicalOperator
@property
def input_dependency(self) -> LogicalOperator:
    return self._input_dependencies[0]