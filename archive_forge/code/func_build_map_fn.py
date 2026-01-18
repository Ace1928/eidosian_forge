import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.engine import base_preprocessing_layer
from keras.src.engine import functional
from keras.src.engine import sequential
from keras.src.utils import tf_utils
def build_map_fn(node, args, kwargs):
    if not isinstance(args.element_spec, tuple):

        def map_fn(*x):
            return tf.nest.flatten(node.layer(*x, **kwargs))
    else:

        def map_fn(*x):
            return tf.nest.flatten(node.layer(x, **kwargs))
    return map_fn