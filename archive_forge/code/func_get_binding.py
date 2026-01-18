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
def get_binding(self, interface: type) -> Tuple[Binding, 'Binder']:
    is_scope = isinstance(interface, type) and issubclass(interface, Scope)
    is_assisted_builder = _is_specialization(interface, AssistedBuilder)
    try:
        return self._get_binding(interface, only_this_binder=is_scope or is_assisted_builder)
    except (KeyError, UnsatisfiedRequirement):
        if is_scope:
            scope = interface
            self.bind(scope, to=scope(self.injector))
            return self._get_binding(interface)
        if self._auto_bind or self._is_special_interface(interface):
            binding = ImplicitBinding(*self.create_binding(interface))
            self._bindings[interface] = binding
            return (binding, self)
    raise UnsatisfiedRequirement(None, interface)