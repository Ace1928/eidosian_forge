import typing
import types
import inspect
import functools
from . import _uarray
import copyreg
import pickle
import contextlib
from ._uarray import (  # type: ignore
def pickle_state(state):
    return (_uarray._BackendState._unpickle, state._pickle())