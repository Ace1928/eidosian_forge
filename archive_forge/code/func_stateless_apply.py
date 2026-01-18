import tensorflow as tf
from keras.src import backend
from keras.src.backend.common import KerasVariable
from keras.src.optimizers import base_optimizer
def stateless_apply(self, optimizer_variables, grads, trainable_variables):
    raise ValueError('stateless_apply is not supported with the TensorFlow backend (as it is incompatible with tf.distribute).')