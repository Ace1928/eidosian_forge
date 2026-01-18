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
def clone_traits(self, traits=None, memo=None, copy=None, **metadata):
    """ Clones a new object from this one, optionally copying only a
        specified set of traits.

        Creates a new object that is a clone of the current object. If *traits*
        is None (the default), then all explicit trait attributes defined
        for this object are cloned. If *traits* is 'all' or an empty list, the
        list of traits returned by all_trait_names() is used; otherwise,
        *traits* must be a list of the names of the trait attributes to be
        cloned.

        Parameters
        ----------
        traits : list of strings
            The list of names of the trait attributes to copy.
        memo : dict
            A dictionary of objects that have already been copied.
        copy : str
            The type of copy ``deep`` or ``shallow`` to perform on any trait
            that does not have explicit 'copy' metadata. A value of None means
            'copy reference'.

        Returns
        -------
        new :
            The newly cloned object.
        """
    if memo is None:
        memo = {}
    if traits is None:
        traits = self.copyable_trait_names(**metadata)
    elif traits == 'all' or len(traits) == 0:
        traits = self.all_trait_names()
        memo['traits_to_copy'] = 'all'
    memo['traits_copy_mode'] = copy
    new = self.__new__(self.__class__)
    memo[id(self)] = new
    new._init_trait_listeners()
    new._init_trait_observers()
    new.copy_traits(self, traits, memo, copy, **metadata)
    new._post_init_trait_listeners()
    new._post_init_trait_observers()
    new.traits_init()
    new._trait_set_inited()
    return new