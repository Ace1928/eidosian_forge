from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_tpu_ops
from tensorflow.python.ops.gen_tpu_ops import *
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_function
from tensorflow.python.util.tf_export import tf_export
def enqueue_tpu_embedding_sparse_batch(sample_indices, embedding_indices, aggregation_weights, device_ordinal, combiners=None, mode_override=None, name=None):
    """A placeholder op for enqueueing embedding IDs to the TPU.

  Args:
    sample_indices: A list of rank 1 Tensors specifying the training example and
      feature to which the corresponding embedding_indices and
      aggregation_weights values belong. sample_indices[i] must equal b * nf +
      f, where nf is the number of features from the corresponding table, f is
      in [0, nf), and b is in [0, batch size). Both int32 and int64 are allowed,
      and will be converted to int32 internally.
    embedding_indices: A list of rank 1 Tensors, indices into the embedding
      tables. Both int32 and int64 are allowed and will be converted to int32
      internally.
    aggregation_weights: A list of rank 1 Tensors containing per sample -- i.e.,
      per (training example, feature) -- aggregation weights. Both float32 and
      float64 are allowed and will be converted to float32 internally.
    device_ordinal: The TPU device to use. Should be >= 0 and less than the
      number of TPU cores in the task on which the node is placed.
    combiners: A list of string scalars, one for each embedding table that
      specify how to normalize the embedding activations after weighted
      summation. Supported combiners are 'mean', 'sum', or 'sqrtn'. It is
      invalid to have the sum of the weights be 0 for 'mean' or the sum of the
      squared weights be 0 for 'sqrtn'. If combiners isn't passed, the default
      is to use 'sum' for all tables (optional).
    mode_override: A string input that overrides the mode specified in the
      TPUEmbeddingConfiguration. Supported values are {'unspecified',
      'inference', 'train', 'backward_pass_only'}. When set to 'unspecified',
      the mode set in TPUEmbeddingConfiguration is used, otherwise mode_override
      is used (optional).
    name: A name for the operation (optional).

  Returns:
    An EnqueueTPUEmbeddingSparseBatch operation.
  """
    if mode_override is None:
        mode_override = 'unspecified'
    return gen_tpu_ops.enqueue_tpu_embedding_sparse_batch(sample_indices=sample_indices, embedding_indices=embedding_indices, aggregation_weights=aggregation_weights, device_ordinal=device_ordinal, combiners=combiners, mode_override=mode_override, name=name)