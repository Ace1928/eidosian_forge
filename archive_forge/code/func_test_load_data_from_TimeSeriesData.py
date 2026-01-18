import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.interfaces.model_interface import DynamicModelInterface
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_load_data_from_TimeSeriesData(self):
    m = self._make_model()
    interface = DynamicModelInterface(m, m.time)
    new_A = [1.0, 2.0, 3.0]
    new_input = [4.0, 5.0, 6.0]
    data = TimeSeriesData({m.var[:, 'A']: new_A, m.input[:]: new_input}, m.time)
    interface.load_data(data)
    B_data = [m.var[t, 'B'].value for t in m.time]
    self.assertEqual(B_data, [1.0, 1.1, 1.2])
    for i, t in enumerate(m.time):
        self.assertEqual(m.var[t, 'A'].value, new_A[i])
        self.assertEqual(m.input[t].value, new_input[i])