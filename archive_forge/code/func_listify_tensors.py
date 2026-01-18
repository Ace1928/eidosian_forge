import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.utils import tf_utils
def listify_tensors(x):
    """Convert any tensors or numpy arrays to lists for config serialization."""
    if tf.is_tensor(x):
        x = x.numpy()
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return x