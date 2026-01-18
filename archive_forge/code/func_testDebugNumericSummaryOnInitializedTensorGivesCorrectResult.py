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
def testDebugNumericSummaryOnInitializedTensorGivesCorrectResult(self):
    with session.Session(config=no_rewrite_session_config()) as sess:
        a = variable_v1.VariableV1([np.nan, np.nan, 0.0, 0.0, 0.0, -1.0, -3.0, 3.0, 7.0, -np.inf, -np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.nan, np.nan], dtype=np.float32, name='numeric_summary/a')
        b = variable_v1.VariableV1([0.0] * 18, dtype=np.float32, name='numeric_summary/b')
        c = math_ops.add(a, b, name='numeric_summary/c')
        sess.run(variables.global_variables_initializer())
        _, dump = self._debug_run_and_get_dump(sess, c, debug_ops=['DebugNumericSummary'])
        self.assertTrue(dump.loaded_partition_graphs())
        self.assertAllClose([[1.0, 18.0, 4.0, 2.0, 2.0, 3.0, 2.0, 5.0, -3.0, 7.0, 0.85714286, 8.97959184, 1.0, 1.0, 18.0]], dump.get_tensors('numeric_summary/a/read', 0, 'DebugNumericSummary'))