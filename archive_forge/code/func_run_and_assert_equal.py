import numpy as np
from tensorflow.python.ops import variables
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops
from tensorflow.python.platform import test
from tensorflow.python.util import nest
def run_and_assert_equal(self, targets1, targets2, rtol=0.0001, atol=1e-05):
    outputs = self._run_targets(targets1, targets2)
    outputs = nest.flatten(outputs)
    n = len(outputs) // 2
    for i in range(n):
        if outputs[i + n].dtype != np.object_:
            self.assertAllClose(outputs[i + n], outputs[i], rtol=rtol, atol=atol)
        else:
            self.assertAllEqual(outputs[i + n], outputs[i])