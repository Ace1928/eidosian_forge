from pyomo.dae.flatten import flatten_dae_components
from pyomo.common.modeling import NOTSET
from pyomo.core.base.var import Var
from pyomo.core.base.expression import Expression
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.expr.numeric_expr import value as pyo_value
from pyomo.contrib.mpc.interfaces.load_data import (
from pyomo.contrib.mpc.interfaces.copy_values import copy_values_at_time
from pyomo.contrib.mpc.data.find_nearest_index import find_nearest_index
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid
from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.convert import _process_to_dynamic_data
from pyomo.contrib.mpc.modeling.cost_expressions import (
from pyomo.contrib.mpc.modeling.constraints import get_piecewise_constant_constraints
def shift_values_by_time(self, dt):
    """
        Shift values in time indexed variables by a specified time offset.
        """
    seen = set()
    t0 = self.time.first()
    tf = self.time.last()
    time_map = {}
    time_list = list(self.time)
    for var in self._dae_vars:
        if id(var[tf]) in seen:
            continue
        else:
            seen.add(id(var[tf]))
        new_values = []
        for t in time_list:
            if t not in time_map:
                t_new = t + dt
                idx = find_nearest_index(time_list, t_new, tolerance=None)
                t_new = time_list[idx]
                time_map[t] = t_new
            t_new = time_map[t]
            new_values.append(var[t_new].value)
        for i, t in enumerate(self.time):
            var[t].set_value(new_values[i])