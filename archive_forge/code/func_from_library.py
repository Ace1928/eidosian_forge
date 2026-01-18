import collections
import hashlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import context
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
def from_library(lib):
    """Creates _DefinedFunctions initialized from a FunctionDefLibrary proto.

  This method handles assigning the correct gradient functions to each
  function.

  Args:
    lib: a FunctionDefLibrary

  Returns:
    A list of _DefinedFunctions

  Raises:
    ValueError: `lib` is invalid
  """
    if not lib.function and (not lib.gradient):
        return []
    funcs = {fdef.signature.name: fdef for fdef in lib.function}
    for g in lib.gradient:
        if g.function_name not in funcs:
            raise ValueError(f"FunctionDefLibrary missing '{g.function_name}' FunctionDef\n{lib}")
        if g.gradient_func not in funcs:
            raise ValueError(f"FunctionDefLibrary missing '{g.gradient_func}' FunctionDef\n{lib}")
    func_to_grad = collections.defaultdict(lambda: None)
    grad_to_funcs = collections.defaultdict(list)
    for gdef in lib.gradient:
        func_to_grad[gdef.function_name] = gdef.gradient_func
        grad_to_funcs[gdef.gradient_func].append(gdef.function_name)
    ready = [fdef for fdef in lib.function if func_to_grad[fdef.signature.name] is None]
    if not ready:
        raise ValueError(f'FunctionDefLibrary contains cyclic gradient functions!\n{lib}')
    initialized = {}
    while ready:
        fdef = ready.pop()
        name = fdef.signature.name
        grad = initialized.get(func_to_grad[name])
        if func_to_grad[name]:
            assert grad
        defined_func = _from_definition(fdef, grad_func=grad)
        initialized[name] = defined_func
        ready.extend((funcs[f] for f in grad_to_funcs[name]))
    return initialized.values()