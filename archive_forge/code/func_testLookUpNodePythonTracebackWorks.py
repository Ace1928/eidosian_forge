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
def testLookUpNodePythonTracebackWorks(self):
    with session.Session() as sess:
        u_init = constant_op.constant(10.0)
        u = variable_v1.VariableV1(u_init, name='traceback/u')
        v_init = constant_op.constant(20.0)
        v = variable_v1.VariableV1(v_init, name='traceback/v')
        w = math_ops.multiply(u, v, name='traceback/w')
        sess.run(variables.global_variables_initializer())
        _, dump = self._debug_run_and_get_dump(sess, w)
        with self.assertRaisesRegexp(LookupError, 'Python graph is not available for traceback lookup'):
            dump.node_traceback('traceback/w')
        dump.set_python_graph(sess.graph)
        with self.assertRaisesRegexp(KeyError, 'Cannot find node \\"foo\\" in Python graph'):
            dump.node_traceback('foo')
        traceback = dump.node_traceback('traceback/w')
        self.assertIsInstance(traceback, tuple)
        self.assertGreater(len(traceback), 0)
        for trace in traceback:
            self.assertIsInstance(trace, tuple)
        traceback = dump.node_traceback('traceback/w:0')
        self.assertIsInstance(traceback, tuple)
        self.assertGreater(len(traceback), 0)
        for trace in traceback:
            self.assertIsInstance(trace, tuple)