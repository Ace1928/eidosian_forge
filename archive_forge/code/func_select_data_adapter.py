import abc
import contextlib
import functools
import itertools
import math
import random
import numpy as np
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import dataset_creator
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
def select_data_adapter(x, y):
    """Selects a data adapter than can handle a given x and y."""
    adapter_cls = [cls for cls in ALL_ADAPTER_CLS if cls.can_handle(x, y)]
    if not adapter_cls:
        raise ValueError('Failed to find data adapter that can handle input: {}, {}'.format(_type_name(x), _type_name(y)))
    elif len(adapter_cls) > 1:
        raise RuntimeError('Data adapters should be mutually exclusive for handling inputs. Found multiple adapters {} to handle input: {}, {}'.format(adapter_cls, _type_name(x), _type_name(y)))
    return adapter_cls[0]