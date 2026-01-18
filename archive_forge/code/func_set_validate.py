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
def set_validate(self):
    self.is_mapped = False
    self.has_items = False
    self.reversable = True
    post_setattrs = []
    mapped_handlers = []
    validates = []
    fast_validates = []
    slow_validates = []
    for handler in self.handlers:
        fv = getattr(handler, 'fast_validate', None)
        if fv is not None:
            validates.append(handler.validate)
            if fv[0] == ValidateTrait.complex:
                fast_validates.extend(fv[1])
            else:
                fast_validates.append(fv)
        else:
            slow_validates.append(handler.validate)
        post_setattr = getattr(handler, 'post_setattr', None)
        if post_setattr is not None:
            post_setattrs.append(post_setattr)
        if handler.is_mapped:
            self.is_mapped = True
            mapped_handlers.append(handler)
        else:
            self.reversable = False
        if handler.has_items:
            self.has_items = True
    self.validates = validates
    self.slow_validates = slow_validates
    if self.is_mapped:
        self.mapped_handlers = mapped_handlers
    elif hasattr(self, 'mapped_handlers'):
        del self.mapped_handlers
    if len(fast_validates) > 0:
        if len(slow_validates) > 0:
            fast_validates.append((ValidateTrait.slow, self))
        self.fast_validate = (ValidateTrait.complex, tuple(fast_validates))
    elif hasattr(self, 'fast_validate'):
        del self.fast_validate
    if len(post_setattrs) > 0:
        self.post_setattrs = post_setattrs
        self.post_setattr = self._post_setattr
    elif hasattr(self, 'post_setattr'):
        del self.post_setattr