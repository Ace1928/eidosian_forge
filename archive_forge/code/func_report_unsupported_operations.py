import contextlib
from tensorflow.compiler.jit.ops import xla_ops
from tensorflow.compiler.jit.ops import xla_ops_grad  # pylint: disable=unused-import
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import summary_op_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def report_unsupported_operations(self):
    if self._unsupported_ops:
        op_str = '\n'.join(['  %s (%s)' % (op.type, op.name) for op in self._unsupported_ops[:_MAX_WARNING_LINES]])
        logging.warning('%d unsupported operations found: \n%s', len(self._unsupported_ops), op_str)
        if len(self._unsupported_ops) > _MAX_WARNING_LINES:
            logging.warning('... and %d more', len(self._unsupported_ops) - _MAX_WARNING_LINES)