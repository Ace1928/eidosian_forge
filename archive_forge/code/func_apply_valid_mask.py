from collections.abc import Mapping
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def apply_valid_mask(x):
    x[mask_key] = True
    return x