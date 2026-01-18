import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_dataset_ops
def check_types(component_spec):
    if not isinstance(component_spec, tensor_spec.TensorSpec):
        if isinstance(component_spec, dataset_ops.DatasetSpec):
            raise TypeError('`padded_batch` is not supported for datasets of datasets')
        raise TypeError(f'`padded_batch` is only supported for datasets that produce tensor elements but type spec of elements in the input dataset is not a subclass of TensorSpec: `{component_spec}`.')