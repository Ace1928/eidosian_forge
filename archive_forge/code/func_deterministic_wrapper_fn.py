import inspect
from functools import wraps
from typing import Any, Callable, Optional, Type, Union, get_type_hints
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe
from torch.utils.data.datapipes._typing import _DataPipeMeta
def deterministic_wrapper_fn(self, *args, **kwargs) -> IterDataPipe:
    res = self.deterministic_fn(*args, **kwargs)
    if not isinstance(res, bool):
        raise TypeError(f'deterministic_fn of `non_deterministic` decorator is required to return a boolean value, but {type(res)} is found')
    global _determinism
    if _determinism and res:
        raise TypeError(f"{self.cls.__name__} is non-deterministic with the inputs, but you set 'guaranteed_datapipes_determinism'. You can turn off determinism for this DataPipe if that is acceptable for your application")
    return self.cls(*args, **kwargs)