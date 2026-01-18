from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import structure
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device as framework_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util.tf_export import tf_export
def make_one_shot_iterator(self):
    if self._is_gpu_target:
        raise ValueError('`make_one_shot_iterator` is not compatible with GPU execution. Please use `Dataset.make_initializable_iterator()` instead.')
    else:
        return super(_CopyToDeviceDataset, self).make_one_shot_iterator()