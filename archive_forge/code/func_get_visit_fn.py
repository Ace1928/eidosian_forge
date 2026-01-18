from copy import copy
from enum import Enum
from typing import (
from ..pyutils import inspect, snake_to_camel
from . import ast
from .ast import Node, QUERY_DOCUMENT_KEYS
def get_visit_fn(self, kind: str, is_leaving: bool=False) -> Optional[Callable[..., Optional[VisitorAction]]]:
    """Get the visit function for the given node kind and direction.

        .. deprecated:: 3.2
           Please use ``get_enter_leave_for_kind`` instead. Will be removed in v3.3.
        """
    enter_leave = self.get_enter_leave_for_kind(kind)
    return enter_leave.leave if is_leaving else enter_leave.enter