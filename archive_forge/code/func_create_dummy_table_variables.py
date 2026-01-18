import collections
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu.ops import tpu_ops
def create_dummy_table_variables(tpu_embedding):
    """Create dummy embedding table variables.

  The sole purpose of these dummy variables are to trigger gradient
  calculation wrt them so that the gradients wrt activation can be captured
  and later sent to TPU embedding.

  Args:
    tpu_embedding: TPUEmbedding, dummy table variables will be created for use
      with tpu_embedding.

  Returns:
    A tuple of dummy variables and their initializer.

  Raises:
    RuntimeError: if collection to store gradients already exists and is not
    empty.
  """
    dummy_table_variables = collections.OrderedDict()
    for table_id, table in enumerate(tpu_embedding.table_to_features_dict):
        dummy_table_variables[table] = variable_scope.get_variable('tpu_embedding_dummy_table_variable_{}'.format(table), dtype=dtypes.float32, shape=[1], use_resource=True, trainable=True, collections=['tpu_embedding_dummy_table_variables'])
        g = ops.get_default_graph()
        table_gradients = g.get_collection_ref('tpu_embedding_gradients_table_{}'.format(table_id))
        if table_gradients:
            raise RuntimeError('tpu_embedding_gradients_table_{} is not empty.'.format(table_id))
        num_features = len(tpu_embedding.table_to_features_dict[table])
        table_gradients.extend([None for _ in range(num_features)])
    return (dummy_table_variables, variables.variables_initializer(dummy_table_variables.values(), name='tpu_embedding_dummy_table_variables_init'))