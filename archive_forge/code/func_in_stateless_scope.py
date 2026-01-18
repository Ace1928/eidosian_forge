from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
def in_stateless_scope():
    return global_state.get_global_attribute('stateless_scope') is not None