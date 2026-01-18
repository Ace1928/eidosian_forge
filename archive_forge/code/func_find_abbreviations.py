from collections.abc import Iterable, Mapping
from inspect import signature, Parameter
from inspect import getcallargs
from inspect import getfullargspec as check_argspec
import sys
from IPython import get_ipython
from . import (Widget, ValueWidget, Text,
from IPython.display import display, clear_output
from traitlets import HasTraits, Any, Unicode, observe
from numbers import Real, Integral
from warnings import warn
def find_abbreviations(self, kwargs):
    """Find the abbreviations for the given function and kwargs.
        Return (name, abbrev, default) tuples.
        """
    new_kwargs = []
    try:
        sig = self.signature()
    except (ValueError, TypeError):
        return [(key, value, value) for key, value in kwargs.items()]
    for param in sig.parameters.values():
        for name, value, default in _yield_abbreviations_for_parameter(param, kwargs):
            if value is empty:
                raise ValueError('cannot find widget or abbreviation for argument: {!r}'.format(name))
            new_kwargs.append((name, value, default))
    return new_kwargs