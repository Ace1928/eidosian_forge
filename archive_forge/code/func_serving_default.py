import tensorflow as tf
from keras.src.backend.tensorflow.trackable import KerasAutoTrackable
from keras.src.utils import tf_utils
from keras.src.utils import tracking
@tf.function(input_signature=input_signature)
def serving_default(inputs):
    return self(inputs)