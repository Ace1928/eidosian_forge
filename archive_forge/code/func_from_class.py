import inspect
import os
import sys
import breezy
from . import commands as _mod_commands
from . import errors, help_topics, option
from . import plugin as _mod_plugin
from .i18n import gettext
from .trace import mutter, note
def from_class(self, cls):
    """Get new context with same details but lineno of class in source"""
    try:
        lineno = self._cls_to_lineno[cls.__name__]
    except (AttributeError, KeyError):
        mutter('Definition of %r not found in %r', cls, self.path)
        return self
    return self.__class__(self.path, lineno, (self._cls_to_lineno, self._str_to_lineno))