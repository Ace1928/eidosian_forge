import copy
import itertools
from pyomo.common import DeveloperError
from pyomo.common.collections import Sequence
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.global_set import UnindexedComponent_index
def expanded_keys(self):
    _iter = self.__iter__()
    return (_iter.get_last_index() for _ in _iter)