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
def property_depends_on(dependency, settable=False, flushable=False):
    """ Marks the following method definition as being a "cached property"
        that depends on the specified extended trait names. That is, it is a
        property getter which, for performance reasons, caches its most
        recently computed result in an attribute whose name is of the form:
        *_traits_cache_name*, where *name* is the name of the property. A
        method marked as being a cached property needs only to compute and
        return its result. The @property_depends_on decorator automatically
        wraps the decorated method in cache management code that will cache the
        most recently computed value and flush the cache when any of the
        specified dependencies are modified, thus eliminating the need to write
        boilerplate cache management code explicitly. For example::

            file_name = File
            file_contents = Property

            @property_depends_on( 'file_name' )
            def _get_file_contents(self):
                with open(self.file_name, 'rb') as fh:
                    return fh.read()

        In this example, accessing the *file_contents* trait calls the
        _get_file_contents() method only once each time after the **file_name**
        trait is modified. In all other cases, the cached value
        **_file_contents**, which is maintained by the @cached_property wrapper
        code, is returned.
    """

    def decorator(function):
        name = TraitsCache + function.__name__[5:]

        def wrapper(self):
            result = self.__dict__.get(name, Undefined)
            if result is Undefined:
                self.__dict__[name] = result = function(self)
            return result
        wrapper.cached_property = True
        wrapper.depends_on = dependency
        wrapper.settable = settable
        wrapper.flushable = flushable
        return wrapper
    return decorator