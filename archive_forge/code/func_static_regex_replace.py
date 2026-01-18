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
def static_regex_replace(input: _atypes.TensorFuzzingAnnotation[_atypes.String], pattern: str, rewrite: str, replace_global: bool=True, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """Replaces the match of pattern in input with rewrite.

  It follows the re2 syntax (https://github.com/google/re2/wiki/Syntax)

  Args:
    input: A `Tensor` of type `string`. The text to be processed.
    pattern: A `string`. The regular expression to match the input.
    rewrite: A `string`. The rewrite to be applied to the matched expression.
    replace_global: An optional `bool`. Defaults to `True`.
      If True, the replacement is global, otherwise the replacement
      is done only on the first match.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'StaticRegexReplace', name, input, 'pattern', pattern, 'rewrite', rewrite, 'replace_global', replace_global)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return static_regex_replace_eager_fallback(input, pattern=pattern, rewrite=rewrite, replace_global=replace_global, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    pattern = _execute.make_str(pattern, 'pattern')
    rewrite = _execute.make_str(rewrite, 'rewrite')
    if replace_global is None:
        replace_global = True
    replace_global = _execute.make_bool(replace_global, 'replace_global')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('StaticRegexReplace', input=input, pattern=pattern, rewrite=rewrite, replace_global=replace_global, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('pattern', _op.get_attr('pattern'), 'rewrite', _op.get_attr('rewrite'), 'replace_global', _op._get_attr_bool('replace_global'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StaticRegexReplace', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result