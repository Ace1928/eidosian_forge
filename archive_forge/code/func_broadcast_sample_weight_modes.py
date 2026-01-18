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
def broadcast_sample_weight_modes(target_structure, sample_weight_modes):
    """Match sample_weight_modes structure with output structure."""
    if target_structure is None or not nest.flatten(target_structure):
        return sample_weight_modes
    if isinstance(sample_weight_modes, str):
        if isinstance(target_structure, dict):
            return {key: sample_weight_modes for key in target_structure.keys()}
        return [sample_weight_modes for _ in target_structure]
    if sample_weight_modes:
        try:
            nest.assert_same_structure(training_utils.list_to_tuple(target_structure), training_utils.list_to_tuple(sample_weight_modes))
        except (ValueError, TypeError):
            target_str = str(nest.map_structure(lambda _: '...', target_structure))
            mode_str = str(nest.map_structure(lambda _: '...', sample_weight_modes))
            try:
                sample_weight_modes = nest.pack_sequence_as(target_structure, nest.flatten(sample_weight_modes))
                logging.warning('sample_weight modes were coerced from\n  {}\n    to  \n  {}'.format(target_str, mode_str))
            except (ValueError, TypeError):
                raise ValueError('Unable to match target structure and sample_weight_modes structure:\n  {}\n    to  \n  {}'.format(target_str, mode_str))
    return sample_weight_modes