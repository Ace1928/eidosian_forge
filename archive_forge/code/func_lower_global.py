import builtins
import operator
import inspect
from functools import cached_property
import llvmlite.ir
from numba.core import types, utils, ir, generators, cgutils
from numba.core.errors import (ForbiddenConstruct, LoweringError,
from numba.core.lowering import BaseLower
def lower_global(self, name, value):
    """
        1) Check global scope dictionary.
        2) Check __builtins__.
            2a) is it a dictionary (for non __main__ module)
            2b) is it a module (for __main__ module)
        """
    moddict = self.get_module_dict()
    obj = self.pyapi.dict_getitem(moddict, self._freeze_string(name))
    self.incref(obj)
    try:
        if value in _unsupported_builtins:
            raise ForbiddenConstruct('builtins %s() is not supported' % name, loc=self.loc)
    except TypeError:
        pass
    if hasattr(builtins, name):
        obj_is_null = self.is_null(obj)
        bbelse = self.builder.basic_block
        with self.builder.if_then(obj_is_null):
            mod = self.pyapi.dict_getitem(moddict, self._freeze_string('__builtins__'))
            builtin = self.builtin_lookup(mod, name)
            bbif = self.builder.basic_block
        retval = self.builder.phi(self.pyapi.pyobj)
        retval.add_incoming(obj, bbelse)
        retval.add_incoming(builtin, bbif)
    else:
        retval = obj
        with cgutils.if_unlikely(self.builder, self.is_null(retval)):
            self.pyapi.raise_missing_global_error(name)
            self.return_exception_raised()
    return retval