import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.interfaces.model_interface import DynamicModelInterface
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_get_data_at_time_subset(self):
    m = self._make_model()
    interface = DynamicModelInterface(m, m.time)
    data = interface.get_data_at_time(time=[0, 2])
    pred_data = TimeSeriesData({m.var[:, 'A']: [1.0, 1.2], m.var[:, 'B']: [1.0, 1.2], m.input[:]: [1.0, 0.8]}, [0, 2])
    self.assertEqual(data, pred_data)