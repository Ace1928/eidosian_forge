from collections import namedtuple
from pyomo.core.expr.numvalue import value as pyo_value
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid
from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable, _DynamicDataBase
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.find_nearest_index import (
def extract_variables(self, variables, context=None, copy_values=False):
    """
        Only keep variables specified.

        """
    if copy_values:
        raise NotImplementedError('extract_variables with copy_values=True has not been implemented by %s' % self.__class__)
    data = {}
    if not isinstance(variables, (list, tuple)):
        raise TypeError('extract_values only accepts a list or tuple of variables')
    for var in variables:
        cuid = get_indexed_cuid(var, (self._orig_time_set,), context=context)
        data[cuid] = self._data[cuid]
    return IntervalData(data, self._intervals, time_set=self._orig_time_set)