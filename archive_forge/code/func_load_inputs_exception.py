import pyomo.common.unittest as unittest
import pytest
import pyomo.environ as pyo
import pyomo.dae as dae
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import assert_disjoint_intervals, IntervalData
def load_inputs_exception(self):
    m = self.make_model()
    interface = mpc.DynamicModelInterface(m, m.time)
    inputs = {'_v': {(0, 3): 1.0, (3, 6): 2.0, (6, 9): 3.0}}
    inputs = mpc.IntervalData({'_v': [1.0, 2.0, 3.0]}, [(0, 3), (3, 6), (6, 9)])
    with self.assertRaisesRegex(RuntimeError, 'Cannot find'):
        interface.load_data(inputs)