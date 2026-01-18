from pyomo.common.collections import ComponentMap
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.expression import Expression
from pyomo.core.base.set import Set
from pyomo.contrib.mpc.data.series_data import get_indexed_cuid
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
from pyomo.contrib.mpc.data.convert import interval_to_series, _process_to_dynamic_data
def get_quadratic_penalty_at_time(var, t, setpoint, weight=None):
    if weight is None:
        weight = 1.0
    return weight * (var[t] - setpoint) ** 2