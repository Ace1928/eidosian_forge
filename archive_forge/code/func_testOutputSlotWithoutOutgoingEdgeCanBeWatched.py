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
def testOutputSlotWithoutOutgoingEdgeCanBeWatched(self):
    """Test watching output slots not attached to any outgoing edges."""
    with session.Session(config=no_rewrite_session_config()) as sess:
        u_init_val = np.array([[5.0, 3.0], [-1.0, 0.0]])
        u = constant_op.constant(u_init_val, shape=[2, 2], name='u')
        with ops.control_dependencies([u]):
            z = control_flow_ops.no_op(name='z')
        _, dump = self._debug_run_and_get_dump(sess, z)
        self.assertEqual(1, len(dump.dumped_tensor_data))
        datum = dump.dumped_tensor_data[0]
        self.assertEqual('u', datum.node_name)
        self.assertEqual(0, datum.output_slot)
        self.assertEqual('DebugIdentity', datum.debug_op)
        self.assertAllClose([[5.0, 3.0], [-1.0, 0.0]], datum.get_tensor())