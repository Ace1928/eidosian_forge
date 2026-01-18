from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import utils_impl as utils
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['saved_model.build_signature_def', 'saved_model.signature_def_utils.build_signature_def'])
@deprecation.deprecated_endpoints('saved_model.signature_def_utils.build_signature_def')
def build_signature_def(inputs=None, outputs=None, method_name=None, defaults=None):
    """Utility function to build a SignatureDef protocol buffer.

  Args:
    inputs: Inputs of the SignatureDef defined as a proto map of string to
      tensor info.
    outputs: Outputs of the SignatureDef defined as a proto map of string to
      tensor info.
    method_name: Method name of the SignatureDef as a string.
    defaults: Defaults of the SignatureDef defined as a proto map of string to
      TensorProto.

  Returns:
    A SignatureDef protocol buffer constructed based on the supplied arguments.
  """
    signature_def = meta_graph_pb2.SignatureDef()
    if inputs is not None:
        for item in inputs:
            signature_def.inputs[item].CopyFrom(inputs[item])
    if outputs is not None:
        for item in outputs:
            signature_def.outputs[item].CopyFrom(outputs[item])
    if method_name is not None:
        signature_def.method_name = method_name
    if defaults is not None:
        for arg_name, default in defaults.items():
            if isinstance(default, ops.EagerTensor):
                signature_def.defaults[arg_name].CopyFrom(tensor_util.make_tensor_proto(default.numpy()))
            elif default.op.type == 'Const':
                signature_def.defaults[arg_name].CopyFrom(default.op.get_attr('value'))
            else:
                raise ValueError(f'Unable to convert object {str(default)} of type {type(default)} to TensorProto.')
    return signature_def