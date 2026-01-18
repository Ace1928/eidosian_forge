from collections import namedtuple
from pyomo.core.expr.numvalue import value as pyo_value
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid
from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable, _DynamicDataBase
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.find_nearest_index import (
def get_data_at_interval_indices(self, indices):
    if _is_iterable(indices):
        index_list = list(sorted(indices))
        interval_list = [self._intervals[i] for i in indices]
        data = {cuid: [values[idx] for idx in index_list] for cuid, values in self._data.items()}
        time_set = self._orig_time_set
        return IntervalData(data, interval_list, time_set=time_set)
    else:
        return ScalarData({cuid: values[indices] for cuid, values in self._data.items()})