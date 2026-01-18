from collections import namedtuple
from pyomo.core.expr.numvalue import value as pyo_value
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid
from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable, _DynamicDataBase
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.find_nearest_index import (
def get_intervals(self):
    return self._intervals