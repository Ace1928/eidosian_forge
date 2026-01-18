from typing import Dict, TypeVar, Optional, Union, Any, TYPE_CHECKING
from .registry import (
from lazyops.types.lazydict import LazyDict, RT
def get_or_init(self, name: str, default: Optional[RT]=None) -> RT:
    """
        Get an attribute from the dictionary
        If it does not exist, set it to the default value
        """
    if name not in self._dict:
        if default:
            self._dict[name] = default
        else:
            raise ValueError(f'Default value for {name} is None')
    from lazyops.utils.lazy import lazy_import
    if isinstance(self._dict[name], str):
        self._dict[name] = lazy_import(self._dict[name])
        if self.initialize_objects:
            self._dict[name] = self.obj_initializer(name, self._dict[name])
    elif isinstance(self._dict[name], tuple):
        obj_class, kwargs = self._dict[name]
        if isinstance(obj_class, str):
            obj_class = lazy_import(obj_class)
        for k, v in kwargs.items():
            if callable(v):
                kwargs[k] = v()
        self._dict[name] = self.obj_initializer(name, obj_class, **kwargs)
    elif isinstance(self._dict[name], dict):
        self._dict[name] = self.obj_initializer(name, type(self), **self._dict[name])
    return self._dict[name]