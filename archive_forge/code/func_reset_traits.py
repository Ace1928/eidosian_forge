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
def reset_traits(self, traits=None, **metadata):
    """ Resets some or all of an object's trait attributes to their default
        values.

        Resets each of the traits whose names are specified in the *traits*
        list to their default values. If *traits* is None or omitted, the
        method resets all explicitly-defined object trait attributes to their
        default values. Note that this does not affect wildcard trait
        attributes or trait attributes added via add_trait(), unless they are
        explicitly named in *traits*.

        Parameters
        ----------
        traits : list of strings
            Names of trait attributes to reset.

        Returns
        -------
        unresetable : list of strings
            A list of attributes that the method was unable to reset, which is
            empty if all the attributes were successfully reset.
        """
    unresetable = []
    if traits is None:
        traits = self.trait_names(**metadata)
    for name in traits:
        try:
            delattr(self, name)
        except (AttributeError, TraitError):
            unresetable.append(name)
    return unresetable