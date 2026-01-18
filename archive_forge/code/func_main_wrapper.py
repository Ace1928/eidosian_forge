import atexit
import os
import sys
import tempfile
from absl import app
from absl.testing.absltest import *
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def main_wrapper():
    args = argv
    if args is None:
        args = sys.argv
    return app.run(main=g_main, argv=args)