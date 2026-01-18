import math
from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.feature_column import feature_column_lib as fc_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_replication
def get_feature_key_name(self):
    """get_feature_key_name."""
    if self.is_categorical_column_weighted():
        return self.categorical_column.categorical_column.name
    return self.categorical_column.name