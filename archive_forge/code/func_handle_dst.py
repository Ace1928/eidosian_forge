import re
import string
import weakref
from string import whitespace
from types import MethodType
from .constants import DefaultValue
from .trait_base import Undefined, Uninitialized
from .trait_errors import TraitError
from .trait_notifiers import TraitChangeNotifyWrapper
from .util.weakiddict import WeakIDKeyDict
def handle_dst(self, object, name, old, new):
    """ Handles a trait change for an intermediate link trait when the
            notification is for the final destination trait.
        """
    self.next.unregister(old)
    object, name = self.next.register(new)
    if old is not Uninitialized:
        if object is None:
            raise TraitError('on_trait_change handler signature is incompatible with a change to an intermediate trait')
        wh = self.wrapped_handler_ref()
        if wh is not None:
            wh(object, name, old, getattr(object, name, Undefined))