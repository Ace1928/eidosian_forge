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
def lookup_submodule(self, name):
    if '.' in name:
        name, submodule = name.split('.', 1)
    else:
        submodule = None
    module = self.module_entries.get(name, None)
    if submodule and module is not None:
        module = module.lookup_submodule(submodule)
    return module