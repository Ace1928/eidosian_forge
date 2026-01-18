import os
import warnings
from functools import partial
from math import ceil
from uuid import uuid4
import numpy as np
import pyarrow as pa
from multiprocess import get_context
from .. import config
def minimal_tf_collate_fn(features):
    if isinstance(features, dict):
        return features
    elif config.TF_AVAILABLE:
        import tensorflow as tf
    else:
        raise ImportError('Called a Tensorflow-specific function but Tensorflow is not installed.')
    first = features[0]
    batch = {}
    for k, v in first.items():
        if isinstance(v, np.ndarray):
            batch[k] = np.stack([f[k] for f in features])
        elif isinstance(v, tf.Tensor):
            batch[k] = tf.stack([f[k] for f in features])
        else:
            batch[k] = np.array([f[k] for f in features])
    return batch