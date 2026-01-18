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
def testWatchingOnlyOneOfTwoOutputSlotsDoesNotLeadToCausalityFailure(self):
    with session.Session() as sess:
        x_name = 'oneOfTwoSlots/x'
        u_name = 'oneOfTwoSlots/u'
        v_name = 'oneOfTwoSlots/v'
        w_name = 'oneOfTwoSlots/w'
        y_name = 'oneOfTwoSlots/y'
        x = variable_v1.VariableV1([1, 3, 3, 7], dtype=dtypes.int32, name=x_name)
        sess.run(x.initializer)
        unique_x, indices, _ = array_ops.unique_with_counts(x, name=u_name)
        v = math_ops.add(unique_x, unique_x, name=v_name)
        w = math_ops.add(indices, indices, name=w_name)
        y = math_ops.add(w, w, name=y_name)
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        debug_utils.add_debug_tensor_watch(run_options, u_name, 0, debug_urls=self._debug_urls())
        debug_utils.add_debug_tensor_watch(run_options, w_name, 0, debug_urls=self._debug_urls())
        debug_utils.add_debug_tensor_watch(run_options, y_name, 0, debug_urls=self._debug_urls())
        run_metadata = config_pb2.RunMetadata()
        sess.run([v, y], options=run_options, run_metadata=run_metadata)
        dump = debug_data.DebugDumpDir(self._dump_root, partition_graphs=run_metadata.partition_graphs, validate=True)
        self.assertAllClose([1, 3, 7], dump.get_tensors(u_name, 0, 'DebugIdentity')[0])