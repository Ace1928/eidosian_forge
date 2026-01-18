import pyomo.common.unittest as unittest
import pytest
import pyomo.environ as pyo
import pyomo.dae as dae
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import assert_disjoint_intervals, IntervalData
def test_load_inputs_some_time(self):
    m = self.make_model()
    interface = mpc.DynamicModelInterface(m, m.time)
    inputs = mpc.IntervalData({'v': [1.0]}, [(2, 4)])
    interface.load_data(inputs)
    for t in m.time:
        if t == 3 or t == 4:
            self.assertEqual(m.v[t].value, 1.0)
        else:
            self.assertEqual(m.v[t].value, 0.0)