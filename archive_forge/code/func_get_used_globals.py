from collections import namedtuple, OrderedDict
import dis
import inspect
import itertools
from types import CodeType, ModuleType
from numba.core import errors, utils, serialize
from numba.core.utils import PYVERSION
def get_used_globals(self):
    """
        Get a {name: value} map of the globals used by this code
        object and any nested code objects.
        """
    return self._compute_used_globals(self.func_id.func, self.table, self.co_consts, self.co_names)