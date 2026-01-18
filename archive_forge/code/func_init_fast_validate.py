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
def init_fast_validate(self):
    """ Sets up the C-level fast validator. """
    if self.adapt == 0:
        fast_validate = [ValidateTrait.instance, self.klass]
        if self._allow_none:
            fast_validate = [ValidateTrait.instance, None, self.klass]
        else:
            fast_validate = [ValidateTrait.instance, self.klass]
        if self.klass in TypeTypes:
            fast_validate[0] = ValidateTrait.type
        self.fast_validate = tuple(fast_validate)
    else:
        self.fast_validate = (ValidateTrait.adapt, self.klass, self.adapt, self._allow_none)