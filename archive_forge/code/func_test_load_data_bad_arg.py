import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.interfaces.model_interface import DynamicModelInterface
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_load_data_bad_arg(self):
    m = self._make_model()
    interface = DynamicModelInterface(m, m.time)
    new_A = [1.0, 2.0, 3.0]
    new_input = [4.0, 5.0, 6.0]
    data = ({m.var[:, 'A']: new_A, m.input[:]: new_input}, m.time)
    msg = 'can only be set if data is IntervalData-compatible'
    with self.assertRaisesRegex(RuntimeError, msg):
        interface.load_data(data, prefer_left=True)