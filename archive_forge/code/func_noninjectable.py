import functools
import inspect
import itertools
import logging
import sys
import threading
import types
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import (
def noninjectable(*args: str) -> Callable[[CallableT], CallableT]:
    """Mark some parameters as not injectable.

    This serves as documentation for people reading the code and will prevent
    Injector from ever attempting to provide the parameters.

    For example:

    >>> class Service:
    ...    pass
    ...
    >>> class SomeClass:
    ...     @inject
    ...     @noninjectable('user_id')
    ...     def __init__(self, service: Service, user_id: int):
    ...         # ...
    ...         pass

    :func:`noninjectable` decorations can be stacked on top of
    each other and the order in which a function is decorated with
    :func:`inject` and :func:`noninjectable`
    doesn't matter.

    .. seealso::

        Generic type :data:`NoInject`
            A nicer way to declare parameters as noninjectable.

        Function :func:`get_bindings`
            A way to inspect how various injection declarations interact with each other.

    """

    def decorator(function: CallableT) -> CallableT:
        argspec = inspect.getfullargspec(inspect.unwrap(function))
        for arg in args:
            if arg not in argspec.args and arg not in argspec.kwonlyargs:
                raise UnknownArgument('Unable to mark unknown argument %s as non-injectable.' % arg)
        existing: Set[str] = getattr(function, '__noninjectables__', set())
        merged = existing | set(args)
        cast(Any, function).__noninjectables__ = merged
        return function
    return decorator