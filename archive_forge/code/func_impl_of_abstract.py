import abc
import functools
from typing import cast, Callable, Set, TypeVar
def impl_of_abstract(*args, **kwargs):
    return impl(*args, **kwargs)