from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.util import compat
from tensorflow.python.util._pywrap_checkpoint_reader import CheckpointReader
from tensorflow.python.util.tf_export import tf_export
def has_tensor(self, tensor_str):
    return self._HasTensor(compat.as_bytes(tensor_str))