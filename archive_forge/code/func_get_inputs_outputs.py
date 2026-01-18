from tensorflow.lite.python import util
from tensorflow.lite.python.convert_phase import Component
from tensorflow.lite.python.convert_phase import convert_phase
from tensorflow.lite.python.convert_phase import SubComponent
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader
def get_inputs_outputs(signature_def):
    """Get inputs and outputs from SignatureDef.

  Args:
    signature_def: SignatureDef in the meta_graph_def for conversion.

  Returns:
    The inputs and outputs in the graph for conversion.
  """
    inputs_tensor_info = signature_def.inputs
    outputs_tensor_info = signature_def.outputs

    def gather_names(tensor_info):
        return [tensor_info[key].name for key in tensor_info]
    inputs = gather_names(inputs_tensor_info)
    outputs = gather_names(outputs_tensor_info)
    return (inputs, outputs)