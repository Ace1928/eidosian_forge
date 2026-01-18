import os
import warnings
from functools import partial
from math import ceil
from uuid import uuid4
import numpy as np
import pyarrow as pa
from multiprocess import get_context
from .. import config
def ensure_shapes(input_dict):
    return {key: tf.ensure_shape(val, output_signature[key].shape[1:]) for key, val in input_dict.items()}