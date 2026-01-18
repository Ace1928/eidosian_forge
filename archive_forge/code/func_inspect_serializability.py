import inspect
from contextlib import contextmanager
from typing import Any, Optional, Set, Tuple
import ray  # noqa: F401
import colorama
import ray.cloudpickle as cp
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def inspect_serializability(base_obj: Any, name: Optional[str]=None, depth: int=3, print_file: Optional[Any]=None) -> Tuple[bool, Set[FailureTuple]]:
    """Identifies what objects are preventing serialization.

    Args:
        base_obj: Object to be serialized.
        name: Optional name of string.
        depth: Depth of the scope stack to walk through. Defaults to 3.
        print_file: file argument that will be passed to print().

    Returns:
        bool: True if serializable.
        set[FailureTuple]: Set of unserializable objects.

    .. versionadded:: 1.1.0

    """
    printer = _Printer(print_file)
    return _inspect_serializability(base_obj, name, depth, None, None, printer)