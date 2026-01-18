import logging
import operator
import contextlib
import itertools
from pprint import pprint
from collections import OrderedDict, defaultdict
from functools import reduce
from numba.core import types, utils, typing, ir, config
from numba.core.typing.templates import Signature
from numba.core.errors import (TypingError, UntypedAttributeError,
from numba.core.funcdesc import qualifying_prefix
from numba.core.typeconv import Conversion
def constrain_statement(self, inst):
    if isinstance(inst, ir.Assign):
        self.typeof_assign(inst)
    elif isinstance(inst, ir.SetItem):
        self.typeof_setitem(inst)
    elif isinstance(inst, ir.StaticSetItem):
        self.typeof_static_setitem(inst)
    elif isinstance(inst, ir.DelItem):
        self.typeof_delitem(inst)
    elif isinstance(inst, ir.SetAttr):
        self.typeof_setattr(inst)
    elif isinstance(inst, ir.Print):
        self.typeof_print(inst)
    elif isinstance(inst, ir.StoreMap):
        self.typeof_storemap(inst)
    elif isinstance(inst, (ir.Jump, ir.Branch, ir.Return, ir.Del)):
        pass
    elif isinstance(inst, (ir.DynamicRaise, ir.DynamicTryRaise)):
        pass
    elif isinstance(inst, (ir.StaticRaise, ir.StaticTryRaise)):
        pass
    elif isinstance(inst, ir.PopBlock):
        pass
    elif type(inst) in typeinfer_extensions:
        f = typeinfer_extensions[type(inst)]
        f(inst, self)
    else:
        msg = 'Unsupported constraint encountered: %s' % inst
        raise UnsupportedError(msg, loc=inst.loc)