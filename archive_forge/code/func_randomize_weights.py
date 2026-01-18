import copy
import random
import re
import struct
import sys
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.python.platform import gfile
def randomize_weights(model, random_seed=0, buffers_to_skip=None):
    """Randomize weights in a model.

  Args:
    model: The model in which to randomize weights.
    random_seed: The input to the random number generator (default value is 0).
    buffers_to_skip: The list of buffer indices to skip. The weights in these
      buffers are left unmodified.
  """
    random.seed(random_seed)
    buffers = model.buffers
    buffer_ids = range(1, len(buffers))
    if buffers_to_skip is not None:
        buffer_ids = [idx for idx in buffer_ids if idx not in buffers_to_skip]
    buffer_types = {}
    for graph in model.subgraphs:
        for op in graph.operators:
            if op.inputs is None:
                break
            for input_idx in op.inputs:
                tensor = graph.tensors[input_idx]
                buffer_types[tensor.buffer] = type_to_name(tensor.type)
    for i in buffer_ids:
        buffer_i_data = buffers[i].data
        buffer_i_size = 0 if buffer_i_data is None else buffer_i_data.size
        if buffer_i_size == 0:
            continue
        buffer_type = buffer_types.get(i, 'INT8')
        if buffer_type.startswith('FLOAT'):
            format_code = 'e' if buffer_type == 'FLOAT16' else 'f'
            for offset in range(0, buffer_i_size, struct.calcsize(format_code)):
                value = random.uniform(-0.5, 0.5)
                struct.pack_into(format_code, buffer_i_data, offset, value)
        else:
            for j in range(buffer_i_size):
                buffer_i_data[j] = random.randint(0, 255)