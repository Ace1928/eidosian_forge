import inspect
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
def print_(*objects, **kwargs):
    """Overload of the print builtin."""
    unknown_kwargs = tuple(set(kwargs.keys()) - set(('sep', 'end', 'file', 'flush')))
    if unknown_kwargs:
        raise ValueError('invalid keyword arguments: {}'.format(unknown_kwargs))
    print_fn = _py_print
    for x in objects:
        print_override = registry_lookup(print_registry, x)
        if print_override is not None:
            print_fn = print_override
            break
    if print_fn is _py_print:
        assert not any((tensor_util.is_tf_type(s) for s in objects))
    return print_fn(*objects, **kwargs)