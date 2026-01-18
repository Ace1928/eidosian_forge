from collections import abc
import os
import time
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util.tf_export import tf_export
@tf_export('train.load_variable')
def load_variable(ckpt_dir_or_file, name):
    """Returns the tensor value of the given variable in the checkpoint.

  When the variable name is unknown, you can use `tf.train.list_variables` to
  inspect all the variable names.

  Example usage:

  ```python
  import tensorflow as tf
  a = tf.Variable(1.0)
  b = tf.Variable(2.0)
  ckpt = tf.train.Checkpoint(var_list={'a': a, 'b': b})
  ckpt_path = ckpt.save('tmp-ckpt')
  var= tf.train.load_variable(
      ckpt_path, 'var_list/a/.ATTRIBUTES/VARIABLE_VALUE')
  print(var)  # 1.0
  ```

  Args:
    ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint.
    name: Name of the variable to return.

  Returns:
    A numpy `ndarray` with a copy of the value of this variable.
  """
    if name.endswith(':0'):
        name = name[:-2]
    reader = load_checkpoint(ckpt_dir_or_file)
    return reader.get_tensor(name)