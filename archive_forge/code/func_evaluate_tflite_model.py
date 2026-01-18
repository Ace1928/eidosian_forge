from tensorflow.core.protobuf import config_pb2
from tensorflow.lite.python import interpreter
from tensorflow.lite.python import lite
from tensorflow.python.eager import def_function
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training import saver
def evaluate_tflite_model(tflite_model, input_ndarrays):
    """Evaluates the provided tf.lite model with the given input ndarrays.

  Args:
    tflite_model: bytes. The serialized tf.lite model.
    input_ndarrays: A list of NumPy arrays to feed as input to the model.

  Returns:
    A list of ndarrays produced by the model.

  Raises:
    ValueError: If the number of input arrays does not match the number of
      inputs the model expects.
  """
    the_interpreter = interpreter.Interpreter(model_content=tflite_model)
    the_interpreter.allocate_tensors()
    input_details = the_interpreter.get_input_details()
    output_details = the_interpreter.get_output_details()
    if len(input_details) != len(input_ndarrays):
        raise ValueError('Wrong number of inputs: provided=%s, input_details=%s output_details=%s' % (input_ndarrays, input_details, output_details))
    for input_tensor, data in zip(input_details, input_ndarrays):
        the_interpreter.set_tensor(input_tensor['index'], data)
    the_interpreter.invoke()
    return [the_interpreter.get_tensor(details['index']) for details in output_details]