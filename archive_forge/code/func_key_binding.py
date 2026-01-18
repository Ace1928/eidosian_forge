from __future__ import annotations
from abc import ABCMeta, abstractmethod, abstractproperty
from inspect import isawaitable
from typing import (
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.filters import FilterOrBool, Never, to_filter
from prompt_toolkit.keys import KEY_ALIASES, Keys
def key_binding(filter: FilterOrBool=True, eager: FilterOrBool=False, is_global: FilterOrBool=False, save_before: Callable[[KeyPressEvent], bool]=lambda event: True, record_in_macro: FilterOrBool=True) -> Callable[[KeyHandlerCallable], Binding]:
    """
    Decorator that turn a function into a `Binding` object. This can be added
    to a `KeyBindings` object when a key binding is assigned.
    """
    assert save_before is None or callable(save_before)
    filter = to_filter(filter)
    eager = to_filter(eager)
    is_global = to_filter(is_global)
    save_before = save_before
    record_in_macro = to_filter(record_in_macro)
    keys = ()

    def decorator(function: KeyHandlerCallable) -> Binding:
        return Binding(keys, function, filter=filter, eager=eager, is_global=is_global, save_before=save_before, record_in_macro=record_in_macro)
    return decorator