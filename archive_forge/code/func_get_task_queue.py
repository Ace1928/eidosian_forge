from typing import Dict, TypeVar, Optional, Union, Any, TYPE_CHECKING
from .registry import (
from lazyops.types.lazydict import LazyDict, RT
def get_task_queue(cls, name: str, kind: Optional[str]=None) -> 'TaskQueue':
    """
        Returns a Task Queue

        - Should be overwritten by the subclass
        """
    raise NotImplementedError