from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_config
from tensorflow.python.ops.gen_parsing_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('io.parse_single_sequence_example', v1=['io.parse_single_sequence_example', 'parse_single_sequence_example'])
@dispatch.add_dispatch_support
def parse_single_sequence_example(serialized, context_features=None, sequence_features=None, example_name=None, name=None):
    """Parses a single `SequenceExample` proto.

  Parses a single serialized [`SequenceExample`](https://www.tensorflow.org/code/tensorflow/core/example/example.proto)
  proto given in `serialized`.

  This op parses a serialized sequence example into a tuple of dictionaries,
  each mapping keys to `Tensor` and `SparseTensor` objects.
  The first dictionary contains mappings for keys appearing in
  `context_features`, and the second dictionary contains mappings for keys
  appearing in `sequence_features`.

  At least one of `context_features` and `sequence_features` must be provided
  and non-empty.

  The `context_features` keys are associated with a `SequenceExample` as a
  whole, independent of time / frame.  In contrast, the `sequence_features` keys
  provide a way to access variable-length data within the `FeatureList` section
  of the `SequenceExample` proto.  While the shapes of `context_features` values
  are fixed with respect to frame, the frame dimension (the first dimension)
  of `sequence_features` values may vary between `SequenceExample` protos,
  and even between `feature_list` keys within the same `SequenceExample`.

  `context_features` contains `VarLenFeature`, `RaggedFeature`, and
  `FixedLenFeature` objects. Each `VarLenFeature` is mapped to a `SparseTensor`;
  each `RaggedFeature` is mapped to a `RaggedTensor`; and each `FixedLenFeature`
  is mapped to a `Tensor`, of the specified type, shape, and default value.

  `sequence_features` contains `VarLenFeature`, `RaggedFeature`, and
  `FixedLenSequenceFeature` objects. Each `VarLenFeature` is mapped to a
  `SparseTensor`; each `RaggedFeature` is mapped to a `RaggedTensor`; and each
  `FixedLenSequenceFeature` is mapped to a `Tensor`, each of the specified type.
  The shape will be `(T,) + df.dense_shape` for `FixedLenSequenceFeature` `df`,
  where `T` is the length of the associated `FeatureList` in the
  `SequenceExample`. For instance, `FixedLenSequenceFeature([])` yields a scalar
  1-D `Tensor` of static shape `[None]` and dynamic shape `[T]`, while
  `FixedLenSequenceFeature([k])` (for `int k >= 1`) yields a 2-D matrix `Tensor`
  of static shape `[None, k]` and dynamic shape `[T, k]`.

  Each `SparseTensor` corresponding to `sequence_features` represents a ragged
  vector.  Its indices are `[time, index]`, where `time` is the `FeatureList`
  entry and `index` is the value's index in the list of values associated with
  that time.

  `FixedLenFeature` entries with a `default_value` and `FixedLenSequenceFeature`
  entries with `allow_missing=True` are optional; otherwise, we will fail if
  that `Feature` or `FeatureList` is missing from any example in `serialized`.

  `example_name` may contain a descriptive name for the corresponding serialized
  proto. This may be useful for debugging purposes, but it has no effect on the
  output. If not `None`, `example_name` must be a scalar.

  Note that the batch version of this function, `tf.parse_sequence_example`,
  is written for better memory efficiency and will be faster on large
  `SequenceExample`s.

  Args:
    serialized: A scalar (0-D Tensor) of type string, a single binary
      serialized `SequenceExample` proto.
    context_features: A mapping of feature keys to `FixedLenFeature` or
      `VarLenFeature` or `RaggedFeature` values. These features are associated
      with a `SequenceExample` as a whole.
    sequence_features: A mapping of feature keys to
      `FixedLenSequenceFeature` or `VarLenFeature` or `RaggedFeature` values.
      These features are associated with data within the `FeatureList` section
      of the `SequenceExample` proto.
    example_name: A scalar (0-D Tensor) of strings (optional), the name of
      the serialized proto.
    name: A name for this operation (optional).

  Returns:
    A tuple of two `dict`s, each mapping keys to `Tensor`s and `SparseTensor`s
    and `RaggedTensor`s.

    * The first dict contains the context key/values.
    * The second dict contains the feature_list key/values.

  Raises:
    ValueError: if any feature is invalid.
  """
    if not (context_features or sequence_features):
        raise ValueError('Both context_features and sequence_features are None, but at least one should have values.')
    context_params = _ParseOpParams.from_features(context_features, [VarLenFeature, FixedLenFeature, RaggedFeature])
    feature_list_params = _ParseOpParams.from_features(sequence_features, [VarLenFeature, FixedLenSequenceFeature, RaggedFeature])
    with ops.name_scope(name, 'ParseSingleSequenceExample', [serialized, example_name]):
        context_output, feature_list_output = _parse_single_sequence_example_raw(serialized, context_params, feature_list_params, example_name, name)
        if context_params.ragged_keys:
            context_output = _construct_tensors_for_composite_features(context_features, context_output)
        if feature_list_params.ragged_keys:
            feature_list_output = _construct_tensors_for_composite_features(sequence_features, feature_list_output)
        return (context_output, feature_list_output)