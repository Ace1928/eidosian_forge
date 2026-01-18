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
def testGraphStructureLookupGivesNodeInputsAndRecipients(self):
    u_name, v_name, w_name, dump = self._session_run_for_graph_structure_lookup()
    u_read_name = u_name + '/read'
    self.assertEqual([], dump.node_inputs(u_name))
    self.assertEqual([u_name], dump.node_inputs(u_read_name))
    self.assertEqual([u_read_name] * 2, dump.node_inputs(v_name))
    self.assertEqual([v_name] * 2, dump.node_inputs(w_name))
    self.assertEqual([], dump.node_inputs(u_name, is_control=True))
    self.assertEqual([], dump.node_inputs(u_read_name, is_control=True))
    self.assertEqual([], dump.node_inputs(v_name, is_control=True))
    self.assertEqual([], dump.node_inputs(w_name, is_control=True))
    self.assertTrue(u_read_name in dump.node_recipients(u_name))
    self.assertEqual(2, dump.node_recipients(u_read_name).count(v_name))
    self.assertEqual(2, dump.node_recipients(v_name).count(w_name))
    self.assertEqual([], dump.node_recipients(u_name, is_control=True))
    self.assertEqual([], dump.node_recipients(u_read_name, is_control=True))
    self.assertEqual([], dump.node_recipients(v_name, is_control=True))
    self.assertEqual([], dump.node_recipients(w_name, is_control=True))
    with self.assertRaisesRegexp(ValueError, 'None of the .* device\\(s\\) has a node named '):
        dump.node_inputs(u_name + 'foo')
    with self.assertRaisesRegexp(ValueError, 'None of the .* device\\(s\\) has a node named '):
        dump.node_recipients(u_name + 'foo')
    self.assertEqual([], dump.transitive_inputs(u_name))
    self.assertEqual([u_name], dump.transitive_inputs(u_read_name))
    self.assertEqual(set([u_name, u_read_name]), set(dump.transitive_inputs(v_name)))
    self.assertEqual(set([u_name, u_read_name, v_name]), set(dump.transitive_inputs(w_name)))
    with self.assertRaisesRegexp(ValueError, 'None of the .* device\\(s\\) has a node named '):
        dump.transitive_inputs(u_name + 'foo')