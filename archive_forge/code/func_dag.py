from typing import TYPE_CHECKING, Dict
from .logical_operator import LogicalOperator
from .plan import Plan
@property
def dag(self) -> 'PhysicalOperator':
    """Get the DAG of physical operators."""
    return self._dag