import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.interfaces.model_interface import DynamicModelInterface
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_load_data_at_time_all(self):
    m = self._make_model()
    interface = DynamicModelInterface(m, m.time)
    data = ScalarData({m.var[:, 'A']: 5.5, m.input[:]: 6.6})
    interface.load_data(data)
    B_data = [m.var[t, 'B'].value for t in m.time]
    self.assertEqual(B_data, [1.0, 1.1, 1.2])
    for t in m.time:
        self.assertEqual(m.var[t, 'A'].value, 5.5)
        self.assertEqual(m.input[t].value, 6.6)