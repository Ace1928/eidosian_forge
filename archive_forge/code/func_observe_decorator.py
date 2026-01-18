import abc
import copy as copy_module
import inspect
import os
import pickle
import re
import types
import warnings
import weakref
from types import FunctionType
from . import __version__ as TraitsVersion
from .adaptation.adaptation_error import AdaptationError
from .constants import DefaultValue, TraitKind
from .ctrait import CTrait, __newobj__
from .ctraits import CHasTraits
from .observation import api as observe_api
from .traits import (
from .trait_types import Any, Bool, Disallow, Event, Python, Str
from .trait_notifiers import (
from .trait_base import (
from .trait_errors import TraitError
from .util.deprecated import deprecated
from .util._traitsui_helpers import check_traitsui_major_version
from .trait_converters import check_trait, mapped_trait_for, trait_for
def observe_decorator(handler):
    """ Create input arguments for HasTraits.observe and attach the input
        to the callable.

        The metaclass will then collect this information for calling
        HasTraits.observe with the decorated function.

        Parameters
        ----------
        handler : callable
            Method of a subclass of HasTraits, with signature of the form
            ``my_method(self, event)``.
        """
    handler_signature = inspect.signature(handler)
    try:
        handler_signature.bind('self', 'event')
    except TypeError:
        warnings.warn("Dubious signature for observe-decorated method. The decorated method should be callable with a single positional argument in addition to 'self'. Did you forget to add an 'event' parameter?", UserWarning, stacklevel=2)
    try:
        observe_inputs = handler._observe_inputs
    except AttributeError:
        observe_inputs = []
        handler._observe_inputs = observe_inputs
    observe_input = dict(graphs=graphs, dispatch=dispatch, post_init=post_init, handler_getter=getattr)
    observe_inputs.append(observe_input)
    return handler