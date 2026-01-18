from collections.abc import Sequence
import functools
import sys
from typing import Any, NamedTuple, Optional, Protocol, TypeVar
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
def set_attr(self, func: Any, api_names_attr: str, names: Sequence[str]) -> None:
    setattr(func, api_names_attr, names)