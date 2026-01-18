from typing import Union
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.util import _pywrap_determinism
from tensorflow.python.util import _pywrap_tensor_float_32_execution
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('config.experimental.disable_mlir_bridge')
def disable_mlir_bridge():
    """Disables experimental MLIR-Based TensorFlow Compiler Bridge."""
    context.context().enable_mlir_bridge = False