import numpy
import math
import types as pytypes
import collections
import warnings
import numba
from numba.core.extending import _Intrinsic
from numba.core import types, typing, ir, analysis, postproc, rewrites, config
from numba.core.typing.templates import signature
from numba.core.analysis import (compute_live_map, compute_use_defs,
from numba.core.errors import (TypingError, UnsupportedError,
import copy
def resolve_func_from_module(func_ir, node):
    """
    This returns the python function that is being getattr'd from a module in
    some IR, it resolves import chains/submodules recursively. Should it not be
    possible to find the python function being called None will be returned.

    func_ir - the FunctionIR object
    node - the IR node from which to start resolving (should be a `getattr`).
    """
    getattr_chain = []

    def resolve_mod(mod):
        if getattr(mod, 'op', False) == 'getattr':
            getattr_chain.insert(0, mod.attr)
            try:
                mod = func_ir.get_definition(mod.value)
            except KeyError:
                return None
            return resolve_mod(mod)
        elif isinstance(mod, (ir.Global, ir.FreeVar)):
            if isinstance(mod.value, pytypes.ModuleType):
                return mod
        return None
    mod = resolve_mod(node)
    if mod is not None:
        defn = mod.value
        for x in getattr_chain:
            defn = getattr(defn, x, False)
            if not defn:
                break
        else:
            return defn
    else:
        return None