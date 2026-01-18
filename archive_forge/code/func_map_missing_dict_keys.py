import copy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.keras import losses as losses_mod
from tensorflow.python.keras import metrics as metrics_mod
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def map_missing_dict_keys(y_pred, struct):
    """Replaces missing dict keys in `struct` with `None` placeholders."""
    if not isinstance(y_pred, dict) or not isinstance(struct, dict):
        return struct
    for k in y_pred.keys():
        if k not in struct:
            struct[k] = None
    return struct