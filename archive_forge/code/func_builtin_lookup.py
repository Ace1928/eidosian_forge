import builtins
import operator
import inspect
from functools import cached_property
import llvmlite.ir
from numba.core import types, utils, ir, generators, cgutils
from numba.core.errors import (ForbiddenConstruct, LoweringError,
from numba.core.lowering import BaseLower
def builtin_lookup(self, mod, name):
    """
        Args
        ----
        mod:
            The __builtins__ dictionary or module, as looked up in
            a module's globals.
        name: str
            The object to lookup
        """
    fromdict = self.pyapi.dict_getitem(mod, self._freeze_string(name))
    self.incref(fromdict)
    bbifdict = self.builder.basic_block
    with cgutils.if_unlikely(self.builder, self.is_null(fromdict)):
        frommod = self.pyapi.object_getattr(mod, self._freeze_string(name))
        with cgutils.if_unlikely(self.builder, self.is_null(frommod)):
            self.pyapi.raise_missing_global_error(name)
            self.return_exception_raised()
        bbifmod = self.builder.basic_block
    builtin = self.builder.phi(self.pyapi.pyobj)
    builtin.add_incoming(fromdict, bbifdict)
    builtin.add_incoming(frommod, bbifmod)
    return builtin