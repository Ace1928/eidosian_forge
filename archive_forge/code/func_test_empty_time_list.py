import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.modeling.cost_expressions import (
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_empty_time_list(self):
    m = self._make_model(n_time_points=3)
    A_target = []
    B_target = []
    setpoint = ({m.var[:, 'A']: A_target, m.var[:, 'B']: B_target}, [])
    variables = [pyo.Reference(m.var[:, 'B']), pyo.Reference(m.var[:, 'A'])]
    msg = 'Time sequence.*is empty'
    with self.assertRaisesRegex(ValueError, msg):
        m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint)