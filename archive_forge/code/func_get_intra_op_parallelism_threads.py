from typing import Union
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.util import _pywrap_determinism
from tensorflow.python.util import _pywrap_tensor_float_32_execution
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('config.threading.get_intra_op_parallelism_threads')
def get_intra_op_parallelism_threads():
    """Get number of threads used within an individual op for parallelism.

  Certain operations like matrix multiplication and reductions can utilize
  parallel threads for speed ups. A value of 0 means the system picks an
  appropriate number.

  Returns:
    Number of parallel threads
  """
    return context.context().intra_op_parallelism_threads