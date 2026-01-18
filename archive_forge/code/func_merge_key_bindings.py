from __future__ import annotations
from abc import ABCMeta, abstractmethod, abstractproperty
from inspect import isawaitable
from typing import (
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.filters import FilterOrBool, Never, to_filter
from prompt_toolkit.keys import KEY_ALIASES, Keys
def merge_key_bindings(bindings: Sequence[KeyBindingsBase]) -> _MergedKeyBindings:
    """
    Merge multiple :class:`.Keybinding` objects together.

    Usage::

        bindings = merge_key_bindings([bindings1, bindings2, ...])
    """
    return _MergedKeyBindings(bindings)