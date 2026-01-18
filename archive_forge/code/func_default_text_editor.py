import collections.abc
import datetime
from importlib import import_module
import operator
from os import fspath
from os.path import isfile, isdir
import re
import sys
from types import FunctionType, MethodType, ModuleType
import uuid
import warnings
from .constants import DefaultValue, TraitKind, ValidateTrait
from .trait_base import (
from .trait_converters import trait_from, trait_cast
from .trait_dict_object import TraitDictEvent, TraitDictObject
from .trait_errors import TraitError
from .trait_list_object import TraitListEvent, TraitListObject
from .trait_set_object import TraitSetEvent, TraitSetObject
from .trait_type import (
from .traits import (
from .util.deprecated import deprecated
from .util.import_symbol import import_symbol
from .editor_factories import (
def default_text_editor(trait, type=None):
    """ Return a default text editor for a trait.

    Parameters
    ----------
    trait : TraitType
        The trait we are constructing the editor for.
    type : callable, optional
        A callable (usually a Python type) to use to evaluate the text content
        of the editor and return the correct type of value for the trait.

    Returns
    -------
    TextEditor
        A TraitsUI TextEditor instance for the trait.
    """
    auto_set = trait.auto_set
    if auto_set is None:
        auto_set = True
    enter_set = trait.enter_set or False
    from traitsui.api import TextEditor
    if type is None:
        return TextEditor(auto_set=auto_set, enter_set=enter_set)
    return TextEditor(auto_set=auto_set, enter_set=enter_set, evaluate=type)