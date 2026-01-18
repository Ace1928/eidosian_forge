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
def multiprovider(function: CallableT) -> CallableT:
    """Like :func:`provider`, but for multibindings. Example usage::

        class MyModule(Module):
            @multiprovider
            def provide_strs(self) -> List[str]:
                return ['str1']

        class OtherModule(Module):
            @multiprovider
            def provide_strs_also(self) -> List[str]:
                return ['str2']

        Injector([MyModule, OtherModule]).get(List[str])  # ['str1', 'str2']

    See also: :meth:`Binder.multibind`."""
    _mark_provider_function(function, allow_multi=True)
    return function