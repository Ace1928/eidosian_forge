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
def is_protocol(tp: type, /) -> bool:
    """Return True if the given type is a Protocol.

        Example::

            >>> from typing_extensions import Protocol, is_protocol
            >>> class P(Protocol):
            ...     def a(self) -> str: ...
            ...     b: int
            >>> is_protocol(P)
            True
            >>> is_protocol(int)
            False
        """
    return isinstance(tp, type) and getattr(tp, '_is_protocol', False) and (tp is not Protocol) and (tp is not typing.Protocol)