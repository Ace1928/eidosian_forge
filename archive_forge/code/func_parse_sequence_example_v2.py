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
def parse_sequence_example_v2(serialized: _atypes.TensorFuzzingAnnotation[_atypes.String], debug_name: _atypes.TensorFuzzingAnnotation[_atypes.String], context_sparse_keys: _atypes.TensorFuzzingAnnotation[_atypes.String], context_dense_keys: _atypes.TensorFuzzingAnnotation[_atypes.String], context_ragged_keys: _atypes.TensorFuzzingAnnotation[_atypes.String], feature_list_sparse_keys: _atypes.TensorFuzzingAnnotation[_atypes.String], feature_list_dense_keys: _atypes.TensorFuzzingAnnotation[_atypes.String], feature_list_ragged_keys: _atypes.TensorFuzzingAnnotation[_atypes.String], feature_list_dense_missing_assumed_empty: _atypes.TensorFuzzingAnnotation[_atypes.Bool], context_dense_defaults, Ncontext_sparse: int=0, context_sparse_types=[], context_ragged_value_types=[], context_ragged_split_types=[], context_dense_shapes=[], Nfeature_list_sparse: int=0, Nfeature_list_dense: int=0, feature_list_dense_types=[], feature_list_sparse_types=[], feature_list_ragged_value_types=[], feature_list_ragged_split_types=[], feature_list_dense_shapes=[], name=None):
    """Transforms a vector of tf.io.SequenceExample protos (as strings) into
typed tensors.

  Args:
    serialized: A `Tensor` of type `string`.
      A scalar or vector containing binary serialized SequenceExample protos.
    debug_name: A `Tensor` of type `string`.
      A scalar or vector containing the names of the serialized protos.
      May contain, for example, table key (descriptive) name for the
      corresponding serialized proto.  This is purely useful for debugging
      purposes, and the presence of values here has no effect on the output.
      May also be an empty vector if no name is available.
    context_sparse_keys: A `Tensor` of type `string`.
      The keys expected in the Examples' features associated with context_sparse
      values.
    context_dense_keys: A `Tensor` of type `string`.
      The keys expected in the SequenceExamples' context features associated with
      dense values.
    context_ragged_keys: A `Tensor` of type `string`.
      The keys expected in the Examples' features associated with context_ragged
      values.
    feature_list_sparse_keys: A `Tensor` of type `string`.
      The keys expected in the FeatureLists associated with sparse values.
    feature_list_dense_keys: A `Tensor` of type `string`.
      The keys expected in the SequenceExamples' feature_lists associated
      with lists of dense values.
    feature_list_ragged_keys: A `Tensor` of type `string`.
      The keys expected in the FeatureLists associated with ragged values.
    feature_list_dense_missing_assumed_empty: A `Tensor` of type `bool`.
      A vector corresponding 1:1 with feature_list_dense_keys, indicating which
      features may be missing from the SequenceExamples.  If the associated
      FeatureList is missing, it is treated as empty.
    context_dense_defaults: A list of `Tensor` objects with types from: `float32`, `int64`, `string`.
      A list of Ncontext_dense Tensors (some may be empty).
      context_dense_defaults[j] provides default values
      when the SequenceExample's context map lacks context_dense_key[j].
      If an empty Tensor is provided for context_dense_defaults[j],
      then the Feature context_dense_keys[j] is required.
      The input type is inferred from context_dense_defaults[j], even when it's
      empty.  If context_dense_defaults[j] is not empty, its shape must match
      context_dense_shapes[j].
    Ncontext_sparse: An optional `int` that is `>= 0`. Defaults to `0`.
    context_sparse_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
      A list of Ncontext_sparse types; the data types of data in
      each context Feature given in context_sparse_keys.
      Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
      DT_INT64 (Int64List), and DT_STRING (BytesList).
    context_ragged_value_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
      RaggedTensor.value dtypes for the ragged context features.
    context_ragged_split_types: An optional list of `tf.DTypes` from: `tf.int32, tf.int64`. Defaults to `[]`.
      RaggedTensor.row_split dtypes for the ragged context features.
    context_dense_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      A list of Ncontext_dense shapes; the shapes of data in
      each context Feature given in context_dense_keys.
      The number of elements in the Feature corresponding to context_dense_key[j]
      must always equal context_dense_shapes[j].NumEntries().
      The shape of context_dense_values[j] will match context_dense_shapes[j].
    Nfeature_list_sparse: An optional `int` that is `>= 0`. Defaults to `0`.
    Nfeature_list_dense: An optional `int` that is `>= 0`. Defaults to `0`.
    feature_list_dense_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
    feature_list_sparse_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
      A list of Nfeature_list_sparse types; the data types
      of data in each FeatureList given in feature_list_sparse_keys.
      Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
      DT_INT64 (Int64List), and DT_STRING (BytesList).
    feature_list_ragged_value_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
      RaggedTensor.value dtypes for the ragged FeatureList features.
    feature_list_ragged_split_types: An optional list of `tf.DTypes` from: `tf.int32, tf.int64`. Defaults to `[]`.
      RaggedTensor.row_split dtypes for the ragged FeatureList features.
    feature_list_dense_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      A list of Nfeature_list_dense shapes; the shapes of
      data in each FeatureList given in feature_list_dense_keys.
      The shape of each Feature in the FeatureList corresponding to
      feature_list_dense_key[j] must always equal
      feature_list_dense_shapes[j].NumEntries().
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (context_sparse_indices, context_sparse_values, context_sparse_shapes, context_dense_values, context_ragged_values, context_ragged_row_splits, feature_list_sparse_indices, feature_list_sparse_values, feature_list_sparse_shapes, feature_list_dense_values, feature_list_dense_lengths, feature_list_ragged_values, feature_list_ragged_outer_splits, feature_list_ragged_inner_splits).

    context_sparse_indices: A list of `Ncontext_sparse` `Tensor` objects with type `int64`.
    context_sparse_values: A list of `Tensor` objects of type `context_sparse_types`.
    context_sparse_shapes: A list of `Ncontext_sparse` `Tensor` objects with type `int64`.
    context_dense_values: A list of `Tensor` objects. Has the same type as `context_dense_defaults`.
    context_ragged_values: A list of `Tensor` objects of type `context_ragged_value_types`.
    context_ragged_row_splits: A list of `Tensor` objects of type `context_ragged_split_types`.
    feature_list_sparse_indices: A list of `Nfeature_list_sparse` `Tensor` objects with type `int64`.
    feature_list_sparse_values: A list of `Tensor` objects of type `feature_list_sparse_types`.
    feature_list_sparse_shapes: A list of `Nfeature_list_sparse` `Tensor` objects with type `int64`.
    feature_list_dense_values: A list of `Tensor` objects of type `feature_list_dense_types`.
    feature_list_dense_lengths: A list of `Nfeature_list_dense` `Tensor` objects with type `int64`.
    feature_list_ragged_values: A list of `Tensor` objects of type `feature_list_ragged_value_types`.
    feature_list_ragged_outer_splits: A list of `Tensor` objects of type `feature_list_ragged_split_types`.
    feature_list_ragged_inner_splits: A list of `Tensor` objects of type `feature_list_ragged_split_types`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ParseSequenceExampleV2', name, serialized, debug_name, context_sparse_keys, context_dense_keys, context_ragged_keys, feature_list_sparse_keys, feature_list_dense_keys, feature_list_ragged_keys, feature_list_dense_missing_assumed_empty, context_dense_defaults, 'Ncontext_sparse', Ncontext_sparse, 'context_sparse_types', context_sparse_types, 'context_ragged_value_types', context_ragged_value_types, 'context_ragged_split_types', context_ragged_split_types, 'context_dense_shapes', context_dense_shapes, 'Nfeature_list_sparse', Nfeature_list_sparse, 'Nfeature_list_dense', Nfeature_list_dense, 'feature_list_dense_types', feature_list_dense_types, 'feature_list_sparse_types', feature_list_sparse_types, 'feature_list_ragged_value_types', feature_list_ragged_value_types, 'feature_list_ragged_split_types', feature_list_ragged_split_types, 'feature_list_dense_shapes', feature_list_dense_shapes)
            _result = _ParseSequenceExampleV2Output._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return parse_sequence_example_v2_eager_fallback(serialized, debug_name, context_sparse_keys, context_dense_keys, context_ragged_keys, feature_list_sparse_keys, feature_list_dense_keys, feature_list_ragged_keys, feature_list_dense_missing_assumed_empty, context_dense_defaults, Ncontext_sparse=Ncontext_sparse, context_sparse_types=context_sparse_types, context_ragged_value_types=context_ragged_value_types, context_ragged_split_types=context_ragged_split_types, context_dense_shapes=context_dense_shapes, Nfeature_list_sparse=Nfeature_list_sparse, Nfeature_list_dense=Nfeature_list_dense, feature_list_dense_types=feature_list_dense_types, feature_list_sparse_types=feature_list_sparse_types, feature_list_ragged_value_types=feature_list_ragged_value_types, feature_list_ragged_split_types=feature_list_ragged_split_types, feature_list_dense_shapes=feature_list_dense_shapes, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if Ncontext_sparse is None:
        Ncontext_sparse = 0
    Ncontext_sparse = _execute.make_int(Ncontext_sparse, 'Ncontext_sparse')
    if context_sparse_types is None:
        context_sparse_types = []
    if not isinstance(context_sparse_types, (list, tuple)):
        raise TypeError("Expected list for 'context_sparse_types' argument to 'parse_sequence_example_v2' Op, not %r." % context_sparse_types)
    context_sparse_types = [_execute.make_type(_t, 'context_sparse_types') for _t in context_sparse_types]
    if context_ragged_value_types is None:
        context_ragged_value_types = []
    if not isinstance(context_ragged_value_types, (list, tuple)):
        raise TypeError("Expected list for 'context_ragged_value_types' argument to 'parse_sequence_example_v2' Op, not %r." % context_ragged_value_types)
    context_ragged_value_types = [_execute.make_type(_t, 'context_ragged_value_types') for _t in context_ragged_value_types]
    if context_ragged_split_types is None:
        context_ragged_split_types = []
    if not isinstance(context_ragged_split_types, (list, tuple)):
        raise TypeError("Expected list for 'context_ragged_split_types' argument to 'parse_sequence_example_v2' Op, not %r." % context_ragged_split_types)
    context_ragged_split_types = [_execute.make_type(_t, 'context_ragged_split_types') for _t in context_ragged_split_types]
    if context_dense_shapes is None:
        context_dense_shapes = []
    if not isinstance(context_dense_shapes, (list, tuple)):
        raise TypeError("Expected list for 'context_dense_shapes' argument to 'parse_sequence_example_v2' Op, not %r." % context_dense_shapes)
    context_dense_shapes = [_execute.make_shape(_s, 'context_dense_shapes') for _s in context_dense_shapes]
    if Nfeature_list_sparse is None:
        Nfeature_list_sparse = 0
    Nfeature_list_sparse = _execute.make_int(Nfeature_list_sparse, 'Nfeature_list_sparse')
    if Nfeature_list_dense is None:
        Nfeature_list_dense = 0
    Nfeature_list_dense = _execute.make_int(Nfeature_list_dense, 'Nfeature_list_dense')
    if feature_list_dense_types is None:
        feature_list_dense_types = []
    if not isinstance(feature_list_dense_types, (list, tuple)):
        raise TypeError("Expected list for 'feature_list_dense_types' argument to 'parse_sequence_example_v2' Op, not %r." % feature_list_dense_types)
    feature_list_dense_types = [_execute.make_type(_t, 'feature_list_dense_types') for _t in feature_list_dense_types]
    if feature_list_sparse_types is None:
        feature_list_sparse_types = []
    if not isinstance(feature_list_sparse_types, (list, tuple)):
        raise TypeError("Expected list for 'feature_list_sparse_types' argument to 'parse_sequence_example_v2' Op, not %r." % feature_list_sparse_types)
    feature_list_sparse_types = [_execute.make_type(_t, 'feature_list_sparse_types') for _t in feature_list_sparse_types]
    if feature_list_ragged_value_types is None:
        feature_list_ragged_value_types = []
    if not isinstance(feature_list_ragged_value_types, (list, tuple)):
        raise TypeError("Expected list for 'feature_list_ragged_value_types' argument to 'parse_sequence_example_v2' Op, not %r." % feature_list_ragged_value_types)
    feature_list_ragged_value_types = [_execute.make_type(_t, 'feature_list_ragged_value_types') for _t in feature_list_ragged_value_types]
    if feature_list_ragged_split_types is None:
        feature_list_ragged_split_types = []
    if not isinstance(feature_list_ragged_split_types, (list, tuple)):
        raise TypeError("Expected list for 'feature_list_ragged_split_types' argument to 'parse_sequence_example_v2' Op, not %r." % feature_list_ragged_split_types)
    feature_list_ragged_split_types = [_execute.make_type(_t, 'feature_list_ragged_split_types') for _t in feature_list_ragged_split_types]
    if feature_list_dense_shapes is None:
        feature_list_dense_shapes = []
    if not isinstance(feature_list_dense_shapes, (list, tuple)):
        raise TypeError("Expected list for 'feature_list_dense_shapes' argument to 'parse_sequence_example_v2' Op, not %r." % feature_list_dense_shapes)
    feature_list_dense_shapes = [_execute.make_shape(_s, 'feature_list_dense_shapes') for _s in feature_list_dense_shapes]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ParseSequenceExampleV2', serialized=serialized, debug_name=debug_name, context_sparse_keys=context_sparse_keys, context_dense_keys=context_dense_keys, context_ragged_keys=context_ragged_keys, feature_list_sparse_keys=feature_list_sparse_keys, feature_list_dense_keys=feature_list_dense_keys, feature_list_ragged_keys=feature_list_ragged_keys, feature_list_dense_missing_assumed_empty=feature_list_dense_missing_assumed_empty, context_dense_defaults=context_dense_defaults, Ncontext_sparse=Ncontext_sparse, context_sparse_types=context_sparse_types, context_ragged_value_types=context_ragged_value_types, context_ragged_split_types=context_ragged_split_types, context_dense_shapes=context_dense_shapes, Nfeature_list_sparse=Nfeature_list_sparse, Nfeature_list_dense=Nfeature_list_dense, feature_list_dense_types=feature_list_dense_types, feature_list_sparse_types=feature_list_sparse_types, feature_list_ragged_value_types=feature_list_ragged_value_types, feature_list_ragged_split_types=feature_list_ragged_split_types, feature_list_dense_shapes=feature_list_dense_shapes, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Ncontext_sparse', _op._get_attr_int('Ncontext_sparse'), 'Tcontext_dense', _op.get_attr('Tcontext_dense'), 'context_sparse_types', _op.get_attr('context_sparse_types'), 'context_ragged_value_types', _op.get_attr('context_ragged_value_types'), 'context_ragged_split_types', _op.get_attr('context_ragged_split_types'), 'context_dense_shapes', _op.get_attr('context_dense_shapes'), 'Nfeature_list_sparse', _op._get_attr_int('Nfeature_list_sparse'), 'Nfeature_list_dense', _op._get_attr_int('Nfeature_list_dense'), 'feature_list_dense_types', _op.get_attr('feature_list_dense_types'), 'feature_list_sparse_types', _op.get_attr('feature_list_sparse_types'), 'feature_list_ragged_value_types', _op.get_attr('feature_list_ragged_value_types'), 'feature_list_ragged_split_types', _op.get_attr('feature_list_ragged_split_types'), 'feature_list_dense_shapes', _op.get_attr('feature_list_dense_shapes'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ParseSequenceExampleV2', _inputs_flat, _attrs, _result)
    _result = [_result[:Ncontext_sparse]] + _result[Ncontext_sparse:]
    _result = _result[:1] + [_result[1:1 + len(context_sparse_types)]] + _result[1 + len(context_sparse_types):]
    _result = _result[:2] + [_result[2:2 + Ncontext_sparse]] + _result[2 + Ncontext_sparse:]
    _result = _result[:3] + [_result[3:3 + len(context_dense_defaults)]] + _result[3 + len(context_dense_defaults):]
    _result = _result[:4] + [_result[4:4 + len(context_ragged_value_types)]] + _result[4 + len(context_ragged_value_types):]
    _result = _result[:5] + [_result[5:5 + len(context_ragged_split_types)]] + _result[5 + len(context_ragged_split_types):]
    _result = _result[:6] + [_result[6:6 + Nfeature_list_sparse]] + _result[6 + Nfeature_list_sparse:]
    _result = _result[:7] + [_result[7:7 + len(feature_list_sparse_types)]] + _result[7 + len(feature_list_sparse_types):]
    _result = _result[:8] + [_result[8:8 + Nfeature_list_sparse]] + _result[8 + Nfeature_list_sparse:]
    _result = _result[:9] + [_result[9:9 + len(feature_list_dense_types)]] + _result[9 + len(feature_list_dense_types):]
    _result = _result[:10] + [_result[10:10 + Nfeature_list_dense]] + _result[10 + Nfeature_list_dense:]
    _result = _result[:11] + [_result[11:11 + len(feature_list_ragged_value_types)]] + _result[11 + len(feature_list_ragged_value_types):]
    _result = _result[:12] + [_result[12:12 + len(feature_list_ragged_split_types)]] + _result[12 + len(feature_list_ragged_split_types):]
    _result = _result[:13] + [_result[13:]]
    _result = _ParseSequenceExampleV2Output._make(_result)
    return _result