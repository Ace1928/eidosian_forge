import collections
import functools
import glob
import os
import tempfile
import threading
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
import tensorflow.python.ops.tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
def testDebugNumericSummaryOnUninitializedTensorGivesCorrectResult(self):
    with session.Session() as sess:
        a = variable_v1.VariableV1([42], dtype=np.float32, name='numeric_summary_uninit/a')
        _, dump = self._debug_run_and_get_dump(sess, a.initializer, debug_ops=['DebugNumericSummary'])
        self.assertTrue(dump.loaded_partition_graphs())
        numeric_summary = dump.get_tensors('numeric_summary_uninit/a', 0, 'DebugNumericSummary')[0]
        self.assertAllClose([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], numeric_summary[0:8])
        self.assertAllClose([1.0, 1.0, 1.0], numeric_summary[12:])
        self.assertTrue(np.isinf(numeric_summary[8]))
        self.assertGreater(numeric_summary[8], 0.0)
        self.assertTrue(np.isinf(numeric_summary[9]))
        self.assertLess(numeric_summary[9], 0.0)
        self.assertTrue(np.isnan(numeric_summary[10]))
        self.assertTrue(np.isnan(numeric_summary[11]))