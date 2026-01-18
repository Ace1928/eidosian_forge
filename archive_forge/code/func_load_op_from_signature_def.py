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
def load_op_from_signature_def(signature_def, key, import_scope=None):
    """Load an Op from a SignatureDef created by op_signature_def().

  Args:
    signature_def: a SignatureDef proto
    key: string key to op in the SignatureDef outputs.
    import_scope: Scope used to import the op

  Returns:
    Op (or possibly Tensor) in the graph with the same name as saved in the
      SignatureDef.

  Raises:
    NotFoundError: If the op could not be found in the graph.
  """
    tensor_info = signature_def.outputs[key]
    try:
        return utils.get_element_from_tensor_info(tensor_info, import_scope=import_scope)
    except KeyError:
        raise errors.NotFoundError(None, None, f'The key "{key}" could not be found in the graph. Please make sure the SavedModel was created by the internal _SavedModelBuilder. If you are using the public API, please make sure the SignatureDef in the SavedModel does not contain the key "{key}".')