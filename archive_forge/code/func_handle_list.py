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
def handle_list(self, object, name, old, new):
    """ Handles a trait change for a list (or set) trait.
        """
    if old is not None and old is not Uninitialized:
        unregister = self.next.unregister
        for obj in old:
            unregister(obj)
    register = self.next.register
    for obj in new:
        register(obj)