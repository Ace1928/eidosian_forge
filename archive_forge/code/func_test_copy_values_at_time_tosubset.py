import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.interfaces.model_interface import DynamicModelInterface
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_copy_values_at_time_tosubset(self):
    m = self._make_model()
    tf = m.time.last()
    interface = DynamicModelInterface(m, m.time)
    target_points = [t for t in m.time if t != m.time.first()]
    target_subset = set(target_points)
    interface.copy_values_at_time(source_time=tf, target_time=target_points)
    for t in m.time:
        if t in target_subset:
            self.assertEqual(m.var[t, 'A'].value, 1.2)
            self.assertEqual(m.var[t, 'B'].value, 1.2)
            self.assertEqual(m.input[t].value, 0.8)
        else:
            self.assertEqual(m.var[t, 'A'].value, 1.0)
            self.assertEqual(m.var[t, 'B'].value, 1.0)
            self.assertEqual(m.input[t].value, 1.0)