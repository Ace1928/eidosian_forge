import collections
import copy
import json
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util import nest
def map_arguments(self, tensor_dict):
    """Maps Keras Tensors to computed Tensors using `tensor_dict`."""
    if self._single_positional_tensor_passed:
        kt_id, _ = self._keras_inputs_ids_and_indices[0]
        return ((tensor_dict[kt_id].pop(),), {})
    else:
        flat_arguments = copy.copy(self._flat_arguments)
        for kt_id, kt_index in self._keras_inputs_ids_and_indices:
            flat_arguments[kt_index] = tensor_dict[kt_id].pop()
        args, kwargs = nest.pack_sequence_as((self.call_args, self.call_kwargs), flat_arguments)
        return (args, kwargs)