import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.interfaces.model_interface import DynamicModelInterface
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_copy_values_at_time_default(self):
    m = self._make_model()
    interface = DynamicModelInterface(m, m.time)
    interface.copy_values_at_time()
    for t in m.time:
        self.assertEqual(m.var[t, 'A'].value, 1.0)
        self.assertEqual(m.var[t, 'B'].value, 1.0)
        self.assertEqual(m.input[t].value, 1.0)