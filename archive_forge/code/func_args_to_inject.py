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
@private
@synchronized(lock)
def args_to_inject(self, function: Callable, bindings: Dict[str, type], owner_key: object) -> Dict[str, Any]:
    """Inject arguments into a function.

        :param function: The function.
        :param bindings: Map of argument name to binding key to inject.
        :param owner_key: A key uniquely identifying the *scope* of this function.
            For a method this will be the owning class.
        :returns: Dictionary of resolved arguments.
        """
    dependencies = {}
    key = (owner_key, function, tuple(sorted(bindings.items())))

    def repr_key(k: Tuple[object, Callable, Tuple[Tuple[str, type], ...]]) -> str:
        owner_key, function, bindings = k
        return '%s.%s(injecting %s)' % (tuple(map(_describe, k[:2])) + (dict(k[2]),))
    log.debug('%sProviding %r for %r', self._log_prefix, bindings, function)
    if key in self._stack:
        raise CircularDependency('circular dependency detected: %s -> %s' % (' -> '.join(map(repr_key, self._stack)), repr_key(key)))
    self._stack += (key,)
    try:
        for arg, interface in bindings.items():
            try:
                instance: Any = self.get(interface)
            except UnsatisfiedRequirement as e:
                if not e.owner:
                    e = UnsatisfiedRequirement(owner_key, e.interface)
                raise e
            dependencies[arg] = instance
    finally:
        self._stack = tuple(self._stack[:-1])
    return dependencies