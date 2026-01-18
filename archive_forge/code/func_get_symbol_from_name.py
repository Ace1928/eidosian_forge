from collections.abc import Sequence
import functools
import sys
from typing import Any, NamedTuple, Optional, Protocol, TypeVar
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
def get_symbol_from_name(name: str) -> Optional[Any]:
    return _NAME_TO_SYMBOL_MAPPING.get(name)