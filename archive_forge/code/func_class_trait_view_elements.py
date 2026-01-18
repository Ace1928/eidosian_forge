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
@classmethod
def class_trait_view_elements(cls):
    """ Returns the ViewElements object associated with the class.

        The returned object can be used to access all the view elements
        associated with the class.
        """
    view_elements = cls.__dict__[ViewTraits]
    if isinstance(view_elements, dict):
        view_elements = cls._init_trait_view_elements()
    return view_elements