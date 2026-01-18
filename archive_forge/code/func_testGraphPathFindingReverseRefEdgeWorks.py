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
def testGraphPathFindingReverseRefEdgeWorks(self):
    with session.Session(config=no_rewrite_session_config()) as sess:
        v = variable_v1.VariableV1(10.0, name='v')
        delta = variable_v1.VariableV1(1.0, name='delta')
        inc_v = state_ops.assign_add(v, delta, name='inc_v')
        sess.run(variables.global_variables_initializer())
        _, dump = self._debug_run_and_get_dump(sess, inc_v)
        self.assertEqual(['delta', 'delta/read', 'inc_v', 'v'], dump.find_some_path('delta', 'v', include_reversed_ref=True))
        self.assertIsNone(dump.find_some_path('delta', 'v'))