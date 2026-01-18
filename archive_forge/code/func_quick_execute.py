from google.protobuf import text_format
from tensorflow.core.framework import tensor_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import core
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
def quick_execute(op_name, num_outputs, inputs, attrs, ctx, name=None):
    """Execute a TensorFlow operation.

  Args:
    op_name: Name of the TensorFlow operation (see REGISTER_OP in C++ code) to
      execute.
    num_outputs: The number of outputs of the operation to fetch. (Explicitly
      provided instead of being inferred for performance reasons).
    inputs: A list of inputs to the operation. Each entry should be a Tensor, or
      a value which can be passed to the Tensor constructor to create one.
    attrs: A tuple with alternating string attr names and attr values for this
      operation.
    ctx: The value of context.context().
    name: Customized name for the operation.

  Returns:
    List of output Tensor objects. The list is empty if there are no outputs

  Raises:
    An exception on error.
  """
    device_name = ctx.device_name
    try:
        ctx.ensure_initialized()
        inputs = [tensor_conversion_registry.convert(t) if isinstance(t, core_types.Tensor) else t for t in inputs]
        tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name, inputs, attrs, num_outputs)
    except core._NotOkStatusException as e:
        if name is not None:
            e.message += ' name: ' + name
        raise core._status_to_exception(e) from None
    except TypeError as e:
        keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
        if keras_symbolic_tensors:
            raise core._SymbolicException('Inputs to eager execution function cannot be Keras symbolic tensors, but found {}'.format(keras_symbolic_tensors))
        raise e
    return tensors