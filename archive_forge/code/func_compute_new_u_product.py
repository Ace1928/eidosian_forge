import tensorflow.compat.v2 as tf
from keras.src.optimizers import optimizer
from keras.src.saving.object_registration import register_keras_serializable
from tensorflow.python.util.tf_export import keras_export
def compute_new_u_product():
    u_product_t = self._u_product * u_t
    self._u_product.assign(u_product_t)
    self._u_product_counter += 1
    return u_product_t