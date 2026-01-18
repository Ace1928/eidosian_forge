import os
import warnings
from functools import partial
from math import ceil
from uuid import uuid4
import numpy as np
import pyarrow as pa
from multiprocess import get_context
from .. import config
@tf.function(input_signature=[tf.TensorSpec(None, tf.int64)])
def fetch_function(indices):
    output = tf.py_function(getter_fn, inp=[indices], Tout=tout)
    return {key: output[i] for i, key in enumerate(columns_to_np_types.keys())}