from typing import Any, Iterable, Optional, Text, Union, Dict
from absl import logging
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.tpu import tpu_embedding_base
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.types import core
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@tf_export('tpu.experimental.embedding.serving_embedding_lookup')
def cpu_embedding_lookup(inputs: Any, weights: Optional[Any], tables: Dict[tpu_embedding_v2_utils.TableConfig, tf_variables.Variable], feature_config: Union[tpu_embedding_v2_utils.FeatureConfig, Iterable]) -> Any:
    """Apply standard lookup ops with `tf.tpu.experimental.embedding` configs.

  This function is a utility which allows using the
  `tf.tpu.experimental.embedding` config objects with standard lookup functions.
  This can be used when exporting a model which uses
  `tf.tpu.experimental.embedding.TPUEmbedding` for serving on CPU. In particular
  `tf.tpu.experimental.embedding.TPUEmbedding` only supports lookups on TPUs and
  should not be part of your serving graph.

  Note that TPU specific options (such as `max_sequence_length`) in the
  configuration objects will be ignored.

  In the following example we take a trained model (see the documentation for
  `tf.tpu.experimental.embedding.TPUEmbedding` for the context) and create a
  saved model with a serving function that will perform the embedding lookup and
  pass the results to your model:

  ```python
  model = model_fn(...)
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=1024,
      optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  checkpoint = tf.train.Checkpoint(model=model, embedding=embedding)
  checkpoint.restore(...)

  @tf.function(input_signature=[{'feature_one': tf.TensorSpec(...),
                                 'feature_two': tf.TensorSpec(...),
                                 'feature_three': tf.TensorSpec(...)}])
  def serve_tensors(embedding_features):
    embedded_features = tf.tpu.experimental.embedding.serving_embedding_lookup(
        embedding_features, None, embedding.embedding_tables,
        feature_config)
    return model(embedded_features)

  model.embedding_api = embedding
  tf.saved_model.save(model,
                      export_dir=...,
                      signatures={'serving_default': serve_tensors})

  ```

  NOTE: It's important to assign the embedding API object to a member of your
  model as `tf.saved_model.save` only supports saving variables as one
  `Trackable` object. Since the model's weights are in `model` and the
  embedding table are managed by `embedding`, we assign `embedding` to an
  attribute of `model` so that tf.saved_model.save can find the embedding
  variables.

  NOTE: The same `serve_tensors` function and `tf.saved_model.save` call will
  work directly from training.

  Args:
    inputs: a nested structure of Tensors, SparseTensors or RaggedTensors.
    weights: a nested structure of Tensors, SparseTensors or RaggedTensors or
      None for no weights. If not None, structure must match that of inputs, but
      entries are allowed to be None.
    tables: a dict of mapping TableConfig objects to Variables.
    feature_config: a nested structure of FeatureConfig objects with the same
      structure as inputs.

  Returns:
    A nested structure of Tensors with the same structure as inputs.
  """
    nest.assert_same_structure(inputs, feature_config)
    flat_inputs = nest.flatten(inputs)
    flat_weights = [None] * len(flat_inputs)
    if weights is not None:
        nest.assert_same_structure(inputs, weights)
        flat_weights = nest.flatten(weights)
    flat_features = nest.flatten_with_joined_string_paths(feature_config)
    outputs = []
    for inp, weight, (path, feature) in zip(flat_inputs, flat_weights, flat_features):
        table = tables[feature.table]
        if weight is not None:
            if isinstance(inp, tensor.Tensor):
                raise ValueError('Weight specified for {}, but input is dense.'.format(path))
            elif type(weight) is not type(inp):
                raise ValueError('Weight for {} is of type {} but it does not match type of the input which is {}.'.format(path, type(weight), type(inp)))
            elif feature.max_sequence_length > 0:
                raise ValueError('Weight specified for {}, but this is a sequence feature.'.format(path))
        if isinstance(inp, tensor.Tensor):
            if feature.max_sequence_length > 0:
                raise ValueError('Feature {} is a sequence feature but a dense tensor was passed.'.format(path))
            outputs.append(embedding_ops.embedding_lookup_v2(table, inp))
        elif isinstance(inp, sparse_tensor.SparseTensor):
            outputs.append(_embedding_lookup_for_sparse_tensor(inp, weight, table, feature))
        elif isinstance(inp, ragged_tensor.RaggedTensor):
            outputs.append(_embedding_lookup_for_ragged_tensor(inp, weight, table, feature))
        else:
            raise ValueError('Input {} is type {}. Tensor, SparseTensor or RaggedTensor expected.'.format(path, type(inp)))
    return nest.pack_sequence_as(feature_config, outputs)