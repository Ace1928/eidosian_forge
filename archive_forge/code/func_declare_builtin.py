from __future__ import absolute_import
import re
import copy
import operator
from ..Utils import try_finally_contextmanager
from .Errors import warning, error, InternalError, performance_hint
from .StringEncoding import EncodedString
from . import Options, Naming
from . import PyrexTypes
from .PyrexTypes import py_object_type, unspecified_type
from .TypeSlots import (
from . import Future
from . import Code
def declare_builtin(self, name, pos):
    if not hasattr(builtins, name) and name not in Code.non_portable_builtins_map and (name not in Code.uncachable_builtins):
        if self.has_import_star:
            entry = self.declare_var(name, py_object_type, pos)
            return entry
        else:
            if Options.error_on_unknown_names:
                error(pos, 'undeclared name not builtin: %s' % name)
            else:
                warning(pos, 'undeclared name not builtin: %s' % name, 2)
            entry = self.declare(name, None, py_object_type, pos, 'private')
            entry.is_builtin = 1
            return entry
    if Options.cache_builtins:
        for entry in self.cached_builtins:
            if entry.name == name:
                return entry
    if name == 'globals' and (not self.old_style_globals):
        return self.outer_scope.lookup('__Pyx_Globals')
    else:
        entry = self.declare(None, None, py_object_type, pos, 'private')
    if Options.cache_builtins and name not in Code.uncachable_builtins:
        entry.is_builtin = 1
        entry.is_const = 1
        entry.name = name
        entry.cname = Naming.builtin_prefix + name
        self.cached_builtins.append(entry)
        self.undeclared_cached_builtins.append(entry)
    else:
        entry.is_builtin = 1
        entry.name = name
    entry.qualified_name = self.builtin_scope().qualify_name(name)
    return entry