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
def testSuccessiveDebuggingRunsIncreasesCounters(self):
    """Test repeated Session.run() calls with debugger increments counters."""
    with session.Session() as sess:
        ph = array_ops.placeholder(dtypes.float32, name='successive/ph')
        x = array_ops.transpose(ph, name='mismatch/x')
        y = array_ops.squeeze(ph, name='mismatch/y')
        _, dump1 = self._debug_run_and_get_dump(sess, x, feed_dict={ph: np.array([[7.0, 8.0]])}, global_step=1)
        self.assertEqual(1, dump1.core_metadata.global_step)
        self.assertGreaterEqual(dump1.core_metadata.session_run_index, 0)
        self.assertEqual(0, dump1.core_metadata.executor_step_index)
        self.assertEqual([ph.name], dump1.core_metadata.input_names)
        self.assertEqual([x.name], dump1.core_metadata.output_names)
        self.assertEqual([], dump1.core_metadata.target_nodes)
        file_io.delete_recursively(self._dump_root)
        _, dump2 = self._debug_run_and_get_dump(sess, x, feed_dict={ph: np.array([[7.0, 8.0]])}, global_step=2)
        self.assertEqual(2, dump2.core_metadata.global_step)
        self.assertEqual(dump1.core_metadata.session_run_index + 1, dump2.core_metadata.session_run_index)
        self.assertEqual(dump1.core_metadata.executor_step_index + 1, dump2.core_metadata.executor_step_index)
        self.assertEqual([ph.name], dump2.core_metadata.input_names)
        self.assertEqual([x.name], dump2.core_metadata.output_names)
        self.assertEqual([], dump2.core_metadata.target_nodes)
        file_io.delete_recursively(self._dump_root)
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        debug_utils.watch_graph(run_options, sess.graph, debug_urls=self._debug_urls(), global_step=3)
        _, dump3 = self._debug_run_and_get_dump(sess, y, feed_dict={ph: np.array([[7.0, 8.0]])}, global_step=3)
        self.assertEqual(3, dump3.core_metadata.global_step)
        self.assertEqual(dump2.core_metadata.session_run_index + 1, dump3.core_metadata.session_run_index)
        self.assertEqual(0, dump3.core_metadata.executor_step_index)
        self.assertEqual([ph.name], dump3.core_metadata.input_names)
        self.assertEqual([y.name], dump3.core_metadata.output_names)
        self.assertEqual([], dump3.core_metadata.target_nodes)