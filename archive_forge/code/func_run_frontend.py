import dis
from contextlib import contextmanager
import builtins
import operator
from typing import Iterator
from functools import reduce
from numba.core import (
from numba.core.utils import (
from .rvsdg.bc2rvsdg import (
from .rvsdg.regionpasses import RegionVisitor
def run_frontend(func):
    sig = utils.pySignature.from_callable(func)
    argnames = tuple(sig.parameters)
    rvsdg = build_rvsdg(func.__code__, argnames)
    func_id = bytecode.FunctionIdentity.from_function(func)
    func_ir = rvsdg_to_ir(func_id, rvsdg)
    return func_ir