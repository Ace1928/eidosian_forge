import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.modeling.cost_expressions import (
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_constant_setpoint_with_ScalarData(self):
    m = self._make_model()
    setpoint = ScalarData({m.var[:, 'A']: 0.3, m.var[:, 'B']: 0.4})
    variables = [pyo.Reference(m.var[:, 'B']), pyo.Reference(m.var[:, 'A'])]
    m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint)
    pred_expr = {(i, t): (m.var[t, 'B'] - 0.4) ** 2 if i == 0 else (m.var[t, 'A'] - 0.3) ** 2 for i, t in m.var_set * m.time}
    for t in m.time:
        for i in m.var_set:
            self.assertTrue(compare_expressions(pred_expr[i, t], m.penalty[i, t].expr))
            self.assertEqual(pyo.value(pred_expr[i, t]), pyo.value(m.penalty[i, t]))