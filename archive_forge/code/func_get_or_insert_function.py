import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def get_or_insert_function(module, fnty, name):
    """
    Get the function named *name* with type *fnty* from *module*, or insert it
    if it doesn't exist.
    """
    fn = module.globals.get(name, None)
    if fn is None:
        fn = ir.Function(module, fnty, name)
    return fn