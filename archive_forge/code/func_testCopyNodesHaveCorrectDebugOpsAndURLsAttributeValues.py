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
def testCopyNodesHaveCorrectDebugOpsAndURLsAttributeValues(self):
    with session.Session() as sess:
        u = variable_v1.VariableV1(2.1, name='u')
        v = variable_v1.VariableV1(20.0, name='v')
        w = math_ops.multiply(u, v, name='w')
        sess.run(variables.global_variables_initializer())
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        debug_urls = self._debug_urls()
        debug_utils.add_debug_tensor_watch(run_options, 'u', 0, ['DebugNumericSummary(gated_grpc=True)', 'DebugIdentity'], debug_urls=debug_urls)
        debug_utils.add_debug_tensor_watch(run_options, 'v', 0, ['DebugNumericSummary'], debug_urls=debug_urls)
        run_metadata = config_pb2.RunMetadata()
        r = sess.run(w, options=run_options, run_metadata=run_metadata)
        self.assertAllClose(42.0, r)
        u_copy_node_def = None
        v_copy_node_def = None
        for partition_graph in run_metadata.partition_graphs:
            for node_def in partition_graph.node:
                if debug_graphs.is_copy_node(node_def.name):
                    if node_def.name == '__copy_u_0':
                        u_copy_node_def = node_def
                    elif node_def.name == '__copy_v_0':
                        v_copy_node_def = node_def
        self.assertIsNotNone(u_copy_node_def)
        debug_ops_spec = u_copy_node_def.attr['debug_ops_spec'].list.s
        self.assertEqual(2, len(debug_ops_spec))
        self.assertEqual('DebugNumericSummary;%s;1' % debug_urls[0], debug_ops_spec[0].decode('utf-8'))
        self.assertEqual('DebugIdentity;%s;0' % debug_urls[0], debug_ops_spec[1].decode('utf-8'))
        self.assertIsNotNone(v_copy_node_def)
        debug_ops_spec = v_copy_node_def.attr['debug_ops_spec'].list.s
        self.assertEqual(1, len(debug_ops_spec))
        self.assertEqual('DebugNumericSummary;%s;0' % debug_urls[0], debug_ops_spec[0].decode('utf-8'))