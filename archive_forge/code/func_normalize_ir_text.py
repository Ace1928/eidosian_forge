import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def normalize_ir_text(text):
    """
    Normalize the given string to latin1 compatible encoding that is
    suitable for use in LLVM IR.
    """
    return text.encode('utf8').decode('latin1')