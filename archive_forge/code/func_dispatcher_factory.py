from collections import namedtuple, defaultdict
import logging
import operator
from numba.core.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.core import errors, ir, ir_utils
from numba.core.analysis import compute_use_defs, compute_cfg_from_blocks
from numba.core.utils import PYVERSION
def dispatcher_factory(func_ir, objectmode=False, **kwargs):
    from numba.core.dispatcher import LiftedWith, ObjModeLiftedWith
    myflags = flags.copy()
    if objectmode:
        myflags.enable_looplift = False
        myflags.enable_pyobject = True
        myflags.force_pyobject = True
        myflags.no_cpython_wrapper = False
        cls = ObjModeLiftedWith
    else:
        cls = LiftedWith
    return cls(func_ir, typingctx, targetctx, myflags, locals, **kwargs)