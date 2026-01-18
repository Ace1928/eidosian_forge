import abc
import collections
import contextlib
import functools
import importlib
import subprocess
import typing
import warnings
from typing import Union, Iterable, Dict, Optional, Callable, Type
from qiskit.exceptions import MissingOptionalLibraryError, OptionalDependencyImportWarning
from .classtools import wrap_method
@contextlib.contextmanager
def disable_locally(self):
    """
        Create a context, during which the value of the dependency manager will be ``False``.  This
        means that within the context, any calls to this object will behave as if the dependency is
        not available, including raising errors.  It is valid to call this method whether or not the
        dependency has already been evaluated.  This is most useful in tests.
        """
    previous = self._bool
    self._bool = False
    try:
        yield
    finally:
        self._bool = previous