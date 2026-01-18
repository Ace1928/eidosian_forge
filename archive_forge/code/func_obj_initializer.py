from typing import Dict, TypeVar, Optional, Union, Any, TYPE_CHECKING
from .registry import (
from lazyops.types.lazydict import LazyDict, RT
def obj_initializer(self, name: str, obj: RT, **kwargs) -> RT:
    """
        Returns the object initializer

        - Can be overwritten by the subclass to modify the object initialization
        """
    return obj(**kwargs)