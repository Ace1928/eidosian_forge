from typing import TYPE_CHECKING, Dict
from .logical_operator import LogicalOperator
from .plan import Plan
@property
def op_map(self) -> Dict['PhysicalOperator', LogicalOperator]:
    """
        Get a mapping from physical operators to their corresponding logical operator.
        """
    return self._op_map