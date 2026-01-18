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
def set_notify(self, notify):
    """ Set notify state on this listener.

        Parameters
        ----------
        notify : bool
            True if this listener should notify, else False.
        """
    for item in self.items:
        item.set_notify(notify)