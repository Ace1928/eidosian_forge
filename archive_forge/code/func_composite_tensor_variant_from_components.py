import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
def composite_tensor_variant_from_components(components, metadata: str, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Encodes an `ExtensionType` value into a `variant` scalar Tensor.

  Returns a scalar variant tensor containing a single `CompositeTensorVariant`
  with the specified Tensor components and TypeSpec.

  Args:
    components: A list of `Tensor` objects.
      The component tensors for the extension type value.
    metadata: A `string`.
      String serialization for the TypeSpec.  (Note: the encoding for the TypeSpec
      may change in future versions of TensorFlow.)
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'CompositeTensorVariantFromComponents', name, components, 'metadata', metadata)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return composite_tensor_variant_from_components_eager_fallback(components, metadata=metadata, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    metadata = _execute.make_str(metadata, 'metadata')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('CompositeTensorVariantFromComponents', components=components, metadata=metadata, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('metadata', _op.get_attr('metadata'), 'Tcomponents', _op.get_attr('Tcomponents'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('CompositeTensorVariantFromComponents', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result