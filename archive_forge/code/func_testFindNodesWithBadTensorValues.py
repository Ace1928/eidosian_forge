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
def testFindNodesWithBadTensorValues(self):
    with session.Session() as sess:
        u_name = 'testFindNodesWithBadTensorValues/u'
        v_name = 'testFindNodesWithBadTensorValues/v'
        w_name = 'testFindNodesWithBadTensorValues/w'
        x_name = 'testFindNodesWithBadTensorValues/x'
        y_name = 'testFindNodesWithBadTensorValues/y'
        z_name = 'testFindNodesWithBadTensorValues/z'
        u_init = constant_op.constant([2.0, 4.0])
        u = variable_v1.VariableV1(u_init, name=u_name)
        v_init = constant_op.constant([2.0, 1.0])
        v = variable_v1.VariableV1(v_init, name=v_name)
        w = math_ops.subtract(u, v, name=w_name)
        x = math_ops.div(u, w, name=x_name)
        y = math_ops.multiply(w, x, name=y_name)
        z = math_ops.multiply(y, y, name=z_name)
        u.initializer.run()
        v.initializer.run()
        _, dump = self._debug_run_and_get_dump(sess, z, expected_partition_graph_count=self._expected_partition_graph_count)

        def has_bad_value(_, tensor):
            return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))
        bad_data = dump.find(has_bad_value)
        self.assertLessEqual(3, len(bad_data))
        node_names = [datum.node_name for datum in bad_data]
        self.assertIn(x_name, node_names)
        self.assertIn(y_name, node_names)
        self.assertIn(z_name, node_names)
        first_bad_datum = dump.find(has_bad_value, first_n=1)
        self.assertEqual(1, len(first_bad_datum))