import abc
from typing import Optional
from ray.data._internal.logical.interfaces import LogicalOperator
@property
def can_modify_num_rows(self) -> bool:
    return True