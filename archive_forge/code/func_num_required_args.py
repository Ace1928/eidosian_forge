from functools import reduce, partial
import inspect
import sys
from operator import attrgetter, not_
from importlib import import_module
from types import MethodType
from .utils import no_default
from . import _signatures as _sigs
def num_required_args(func, sigspec=None):
    sigspec, rv = _check_sigspec(sigspec, func, _sigs._num_required_args, func)
    if sigspec is None:
        return rv
    return sum((1 for p in sigspec.parameters.values() if p.default is p.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)))