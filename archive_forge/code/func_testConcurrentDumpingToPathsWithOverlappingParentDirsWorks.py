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
def testConcurrentDumpingToPathsWithOverlappingParentDirsWorks(self):
    results = self._generate_dump_from_simple_addition_graph()
    self.assertTrue(results.dump.loaded_partition_graphs())
    self.assertEqual(-1, results.dump.core_metadata.global_step)
    self.assertGreaterEqual(results.dump.core_metadata.session_run_index, 0)
    self.assertGreaterEqual(results.dump.core_metadata.executor_step_index, 0)
    self.assertEqual([], results.dump.core_metadata.input_names)
    self.assertEqual([results.w.name], results.dump.core_metadata.output_names)
    self.assertEqual([], results.dump.core_metadata.target_nodes)
    self.assertEqual(2, results.dump.size)
    self.assertAllClose([results.u_init_val], results.dump.get_tensors('%s/read' % results.u_name, 0, 'DebugIdentity'))
    self.assertAllClose([results.v_init_val], results.dump.get_tensors('%s/read' % results.v_name, 0, 'DebugIdentity'))
    self.assertGreaterEqual(results.dump.get_rel_timestamps('%s/read' % results.u_name, 0, 'DebugIdentity')[0], 0)
    self.assertGreaterEqual(results.dump.get_rel_timestamps('%s/read' % results.v_name, 0, 'DebugIdentity')[0], 0)
    self.assertGreater(results.dump.get_dump_sizes_bytes('%s/read' % results.u_name, 0, 'DebugIdentity')[0], 0)
    self.assertGreater(results.dump.get_dump_sizes_bytes('%s/read' % results.v_name, 0, 'DebugIdentity')[0], 0)