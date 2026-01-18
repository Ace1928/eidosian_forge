import builtins
import operator
import inspect
from functools import cached_property
import llvmlite.ir
from numba.core import types, utils, ir, generators, cgutils
from numba.core.errors import (ForbiddenConstruct, LoweringError,
from numba.core.lowering import BaseLower
def get_builtin_obj(self, name):
    moddict = self.get_module_dict()
    mod = self.pyapi.dict_getitem(moddict, self._freeze_string('__builtins__'))
    return self.builtin_lookup(mod, name)