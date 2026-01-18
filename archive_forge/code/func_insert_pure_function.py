import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def insert_pure_function(module, fnty, name):
    """
    Insert a pure function (in the functional programming sense) in the
    given module.
    """
    fn = get_or_insert_function(module, fnty, name)
    fn.attributes.add('readonly')
    fn.attributes.add('nounwind')
    return fn