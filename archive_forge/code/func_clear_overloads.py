import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
def clear_overloads():
    """Clear all overloads in the registry."""
    _overload_registry.clear()