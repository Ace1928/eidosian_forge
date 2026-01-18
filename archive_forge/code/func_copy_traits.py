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
def copy_traits(self, other, traits=None, memo=None, copy=None, **metadata):
    """ Copies another object's trait attributes into this one.

        Parameters
        ----------
        other : object
            The object whose trait attribute values should be copied.
        traits : list of strings
            A list of names of trait attributes to copy. If None or
            unspecified, the set of names returned by trait_names() is used.
            If 'all' or an empty list, the set of names returned by
            all_trait_names() is used.
        memo : dict
            A dictionary of objects that have already been copied.
        copy : None | 'deep' | 'shallow'
            The type of copy to perform on any trait that does not have
            explicit 'copy' metadata. A value of None means 'copy reference'.

        Returns
        -------
        unassignable : list of strings
            A list of attributes that the method was unable to copy, which is
            empty if all the attributes were successfully copied.

        """
    if traits is None:
        traits = self.copyable_trait_names(**metadata)
    elif traits == 'all' or len(traits) == 0:
        traits = self.all_trait_names()
        if memo is not None:
            memo['traits_to_copy'] = 'all'
    unassignable = []
    deferred = []
    deep_copy = copy == 'deep'
    shallow_copy = copy == 'shallow'
    for name in traits:
        try:
            trait = self.trait(name)
            if trait.type in DeferredCopy:
                deferred.append(name)
                continue
            base_trait = other.base_trait(name)
            if base_trait.type == 'event':
                continue
            value = getattr(other, name)
            copy_type = base_trait.copy
            if copy_type == 'shallow':
                value = copy_module.copy(value)
            elif copy_type == 'ref':
                pass
            elif copy_type == 'deep' or deep_copy:
                if memo is None:
                    value = copy_module.deepcopy(value)
                else:
                    value = copy_module.deepcopy(value, memo)
            elif shallow_copy:
                value = copy_module.copy(value)
            setattr(self, name, value)
        except:
            unassignable.append(name)
    for name in deferred:
        try:
            value = getattr(other, name)
            copy_type = other.base_trait(name).copy
            if copy_type == 'shallow':
                value = copy_module.copy(value)
            elif copy_type == 'ref':
                pass
            elif copy_type == 'deep' or deep_copy:
                if memo is None:
                    value = copy_module.deepcopy(value)
                else:
                    value = copy_module.deepcopy(value, memo)
            elif shallow_copy:
                value = copy_module.copy(value)
            setattr(self, name, value)
        except:
            unassignable.append(name)
    return unassignable