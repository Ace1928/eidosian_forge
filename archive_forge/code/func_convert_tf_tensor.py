import sys
from keras.src import backend as backend_module
from keras.src.backend.common import global_state
def convert_tf_tensor(outputs, dtype=None):
    if backend_module.backend() != 'tensorflow' and (not in_tf_graph()):
        outputs = backend_module.convert_to_tensor(outputs, dtype=dtype)
    return outputs