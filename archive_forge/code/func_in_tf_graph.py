import sys
from keras.src import backend as backend_module
from keras.src.backend.common import global_state
def in_tf_graph():
    if global_state.get_global_attribute('in_tf_graph_scope', False):
        return True
    if 'tensorflow' in sys.modules:
        from keras.src.utils.module_utils import tensorflow as tf
        return not tf.executing_eagerly()
    return False