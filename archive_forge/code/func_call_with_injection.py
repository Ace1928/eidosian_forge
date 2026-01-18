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
def call_with_injection(self, callable: Callable[..., T], self_: Any=None, args: Any=(), kwargs: Any={}) -> T:
    """Call a callable and provide it's dependencies if needed.

        :param self_: Instance of a class callable belongs to if it's a method,
            None otherwise.
        :param args: Arguments to pass to callable.
        :param kwargs: Keyword arguments to pass to callable.
        :type callable: callable
        :type args: tuple of objects
        :type kwargs: dict of string -> object
        :return: Value returned by callable.
        """
    bindings = get_bindings(callable)
    signature = inspect.signature(callable)
    full_args = args
    if self_ is not None:
        full_args = (self_,) + full_args
    bound_arguments = signature.bind_partial(*full_args)
    needed = dict(((k, v) for k, v in bindings.items() if k not in kwargs and k not in bound_arguments.arguments))
    dependencies = self.args_to_inject(function=callable, bindings=needed, owner_key=self_.__class__ if self_ is not None else callable.__module__)
    dependencies.update(kwargs)
    try:
        return callable(*full_args, **dependencies)
    except TypeError as e:
        reraise(e, CallError(self_, callable, args, dependencies, e, self._stack))
        assert False, 'unreachable'