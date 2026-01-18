from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import tf_export
def convert_to_tensor_v1(value, dtype=None, name=None, preferred_dtype=None, dtype_hint=None) -> tensor_lib.Tensor:
    """Converts the given `value` to a `Tensor` (with the TF1 API)."""
    preferred_dtype = deprecation.deprecated_argument_lookup('dtype_hint', dtype_hint, 'preferred_dtype', preferred_dtype)
    return convert_to_tensor_v2(value, dtype, preferred_dtype, name)