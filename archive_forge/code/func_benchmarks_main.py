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
def benchmarks_main(true_main, argv=None):
    """Run benchmarks as declared in argv.

  Args:
    true_main: True main function to run if benchmarks are not requested.
    argv: the command line arguments (if None, uses sys.argv).
  """
    if argv is None:
        argv = sys.argv
    found_arg = [arg for arg in argv if arg.startswith('--benchmark_filter=') or arg.startswith('-benchmark_filter=')]
    if found_arg:
        argv.remove(found_arg[0])
        regex = found_arg[0].split('=')[1]
        app.run(lambda _: _run_benchmarks(regex), argv=argv)
    else:
        true_main()