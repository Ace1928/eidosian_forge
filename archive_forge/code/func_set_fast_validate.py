from importlib import import_module
import sys
from types import FunctionType, MethodType
from .constants import DefaultValue, ValidateTrait
from .trait_base import (
from .trait_base import RangeTypes  # noqa: F401, used by TraitsUI
from .trait_errors import TraitError
from .trait_dict_object import TraitDictEvent, TraitDictObject
from .trait_converters import trait_from
from .trait_handler import TraitHandler
from .trait_list_object import TraitListEvent, TraitListObject
from .util.deprecated import deprecated
def set_fast_validate(self):
    fast_validate = [ValidateTrait.instance, self.aClass]
    if self._allow_none:
        fast_validate = [ValidateTrait.instance, None, self.aClass]
    if self.aClass in TypeTypes:
        fast_validate[0] = ValidateTrait.type
    self.fast_validate = tuple(fast_validate)