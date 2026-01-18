import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.util import nest
def create_colocated_variable(next_creator, **kwargs):
    kwargs['colocate_with'] = colocate_with
    return next_creator(**kwargs)