from collections import namedtuple, OrderedDict
import dis
import inspect
import itertools
from types import CodeType, ModuleType
from numba.core import errors, utils, serialize
from numba.core.utils import PYVERSION
@property
def ordered_offsets(self):
    if not self._ordered_offsets:
        self._ordered_offsets = [o for o in self.table]
    return self._ordered_offsets