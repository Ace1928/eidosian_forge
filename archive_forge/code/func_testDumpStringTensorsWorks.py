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
def testDumpStringTensorsWorks(self):
    with session.Session(config=no_rewrite_session_config()) as sess:
        str1_init_val = np.array(b'abc')
        str2_init_val = np.array(b'def')
        str1_init = constant_op.constant(str1_init_val)
        str2_init = constant_op.constant(str2_init_val)
        str1_name = 'str1'
        str2_name = 'str2'
        str1 = variable_v1.VariableV1(str1_init, name=str1_name)
        str2 = variable_v1.VariableV1(str2_init, name=str2_name)
        str_concat = math_ops.add(str1, str2, name='str_concat')
        str1.initializer.run()
        str2.initializer.run()
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        debug_urls = self._debug_urls()
        debug_utils.add_debug_tensor_watch(run_options, '%s/read' % str1_name, 0, debug_urls=debug_urls)
        debug_utils.add_debug_tensor_watch(run_options, '%s/read' % str2_name, 0, debug_urls=debug_urls)
        run_metadata = config_pb2.RunMetadata()
        sess.run(str_concat, options=run_options, run_metadata=run_metadata)
        self.assertEqual(1, len(run_metadata.partition_graphs))
        dump = debug_data.DebugDumpDir(self._dump_root, partition_graphs=run_metadata.partition_graphs)
        self.assertIn(str1_name, dump.nodes())
        self.assertIn(str2_name, dump.nodes())
        self.assertEqual(2, dump.size)
        self.assertEqual([str1_init_val], dump.get_tensors('%s/read' % str1_name, 0, 'DebugIdentity'))
        self.assertEqual([str2_init_val], dump.get_tensors('%s/read' % str2_name, 0, 'DebugIdentity'))
        self.assertGreaterEqual(dump.get_rel_timestamps('%s/read' % str1_name, 0, 'DebugIdentity')[0], 0)
        self.assertGreaterEqual(dump.get_rel_timestamps('%s/read' % str2_name, 0, 'DebugIdentity')[0], 0)
        self.assertGreater(dump.get_dump_sizes_bytes('%s/read' % str1_name, 0, 'DebugIdentity')[0], 0)
        self.assertGreater(dump.get_dump_sizes_bytes('%s/read' % str2_name, 0, 'DebugIdentity')[0], 0)