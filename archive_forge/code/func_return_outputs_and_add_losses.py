import itertools
import threading
import types
from tensorflow.python.eager import context
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.util import tf_decorator
def return_outputs_and_add_losses(*args, **kwargs):
    """Returns the outputs from the layer call function, and adds the losses."""
    if return_method:
        args = args[1:]
    outputs, losses = fn(*args, **kwargs)
    layer.add_loss(losses, inputs=True)
    if context.executing_eagerly():
        for i in layer._flatten_layers():
            if i is not layer:
                i._eager_losses = [base_layer_utils.REVIVED_LOSS_PLACEHOLDER]
    return outputs