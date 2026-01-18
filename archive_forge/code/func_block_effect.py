from collections import namedtuple, OrderedDict
import dis
import inspect
import itertools
from types import CodeType, ModuleType
from numba.core import errors, utils, serialize
from numba.core.utils import PYVERSION
@property
def block_effect(self):
    """Effect of the block stack
        Returns +1 (push), 0 (none) or -1 (pop)
        """
    if self.opname.startswith('SETUP_'):
        return 1
    elif self.opname == 'POP_BLOCK':
        return -1
    else:
        return 0