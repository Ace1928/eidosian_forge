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
def testDebugCondWatchingWholeGraphWorks(self):
    with session.Session() as sess:
        x = variable_v1.VariableV1(10.0, name='x')
        y = variable_v1.VariableV1(20.0, name='y')
        cond = tf_cond.cond(x > y, lambda: math_ops.add(x, 1), lambda: math_ops.add(y, 1))
        sess.run(variables.global_variables_initializer())
        cond_result, dump = self._debug_run_and_get_dump(sess, cond)
        self.assertEqual(21, cond_result)
        self.assertAllClose([21.0], dump.get_tensors('cond/Merge', 0, 'DebugIdentity'))