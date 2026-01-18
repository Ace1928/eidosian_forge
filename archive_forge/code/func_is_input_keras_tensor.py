import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import input_layer as input_layer_module
from keras.src.engine import keras_tensor
from keras.src.engine import node as node_module
def is_input_keras_tensor(tensor):
    """Check if tensor is directly generated from `tf.keras.Input`.

    This check is useful when constructing the functional model, since we will
    need to clone Nodes and KerasTensors if the model is building from non input
    tensor.

    Args:
      tensor: A `KerasTensor` as inputs to the functional model.

    Returns:
      bool. Whether the tensor is directly generated from `tf.keras.Input`.

    Raises:
      ValueError: if the tensor is not a KerasTensor instance.
    """
    if not node_module.is_keras_tensor(tensor):
        raise ValueError(_KERAS_TENSOR_TYPE_CHECK_ERROR_MSG.format(tensor))
    return tensor.node.is_input