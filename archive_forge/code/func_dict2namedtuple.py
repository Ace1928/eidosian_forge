from __future__ import annotations
import collections
from typing import TYPE_CHECKING
def dict2namedtuple(*args, **kwargs) -> tuple:
    """
    Helper function to create a class `namedtuple` from a dictionary.

    Examples:
        >>> t = dict2namedtuple(foo=1, bar="hello")
        >>> assert t.foo == 1 and t.bar == "hello"

        >>> t = dict2namedtuple([("foo", 1), ("bar", "hello")])
        >>> assert t[0] == t.foo and t[1] == t.bar

    Warnings:
        - The order of the items in the namedtuple is not deterministic if
          kwargs are used.
          namedtuples, however, should always be accessed by attribute hence
          this behaviour should not represent a serious problem.

        - Don't use this function in code in which memory and performance are
          crucial since a dict is needed to instantiate the tuple!
    """
    d = collections.OrderedDict(*args)
    d.update(**kwargs)
    return collections.namedtuple(typename='dict2namedtuple', field_names=list(d.keys()))(**d)