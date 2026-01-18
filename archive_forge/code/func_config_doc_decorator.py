import ctypes
import json
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Iterator, Optional, cast
from ._typing import _F
from .core import _LIB, _check_call, c_str, py_str
def config_doc_decorator(func: _F) -> _F:
    func.__doc__ = doc_template.format(header=none_to_str(header), extra_note=none_to_str(extra_note)) + none_to_str(parameters) + none_to_str(returns) + none_to_str(common_example) + none_to_str(see_also)

    @wraps(func)
    def wrap(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)
    return cast(_F, wrap)