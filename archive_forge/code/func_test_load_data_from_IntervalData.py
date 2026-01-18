import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.interfaces.model_interface import DynamicModelInterface
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_load_data_from_IntervalData(self):
    m = self._make_model(5)
    interface = DynamicModelInterface(m, m.time)
    new_A = [-1.1, -1.2, -1.3]
    new_input = [3.0, 2.9, 2.8]
    data = IntervalData({m.var[:, 'A']: new_A, m.input[:]: new_input}, [(0.0, 0.0), (0.0, 2.0), (2.0, 4.0)])
    interface.load_data(data)
    B_data = [m.var[t, 'B'].value for t in m.time]
    self.assertEqual(B_data, [1.0, 1.1, 1.2, 1.3, 1.4])
    for t in m.time:
        if t == 0:
            idx = 0
        elif t <= 2.0:
            idx = 1
        elif t <= 4.0:
            idx = 2
        self.assertEqual(m.var[t, 'A'].value, new_A[idx])
        self.assertEqual(m.input[t].value, new_input[idx])