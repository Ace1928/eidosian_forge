import abc
import functools
from typing import cast, Callable, Set, TypeVar
def has_some_implementation(name: str) -> bool:
    if name in implemented_by:
        return True
    try:
        value = getattr(cls, name)
    except AttributeError:
        raise TypeError(f"A method named '{name}' was listed as a possible implementation alternative but it does not exist in the definition of {cls!r}.")
    if getattr(value, '__isabstractmethod__', False):
        return False
    if hasattr(value, '_abstract_alternatives_'):
        return False
    return True