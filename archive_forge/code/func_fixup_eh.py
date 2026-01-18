from collections import namedtuple, OrderedDict
import dis
import inspect
import itertools
from types import CodeType, ModuleType
from numba.core import errors, utils, serialize
from numba.core.utils import PYVERSION
@staticmethod
def fixup_eh(ent):
    out = dis._ExceptionTableEntry(start=ent.start + _FIXED_OFFSET, end=ent.end + _FIXED_OFFSET, target=ent.target + _FIXED_OFFSET, depth=ent.depth, lasti=ent.lasti)
    return out