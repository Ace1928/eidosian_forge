import sys
from google.protobuf import message
from tensorflow.core.profiler import tfprof_options_pb2
from tensorflow.core.profiler import tfprof_output_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util import _pywrap_tfprof as print_mdl
from tensorflow.python.util.tf_export import tf_export
def serialize_to_string(self):
    """Serialize the ProfileProto to a binary string.

      Users can write it to file for offline analysis by tfprof commandline
      or graphical interface.

    Returns:
      ProfileProto binary string.
    """
    return print_mdl.SerializeToString()