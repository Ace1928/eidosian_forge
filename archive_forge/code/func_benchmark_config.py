import math
import numbers
import os
import re
import sys
import time
import types
from absl import app
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.util import test_log_pb2
from tensorflow.python.client import timeline
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export('test.benchmark_config')
def benchmark_config():
    """Returns a tf.compat.v1.ConfigProto for disabling the dependency optimizer.

    Returns:
      A TensorFlow ConfigProto object.
  """
    config = config_pb2.ConfigProto()
    config.graph_options.rewrite_options.dependency_optimization = rewriter_config_pb2.RewriterConfig.OFF
    return config