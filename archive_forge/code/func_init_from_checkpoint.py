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
@tf_export(v1=['train.init_from_checkpoint'])
def init_from_checkpoint(ckpt_dir_or_file, assignment_map):
    """Replaces `tf.Variable` initializers so they load from a checkpoint file.

  @compatibility(TF2)
  `tf.compat.v1.train.init_from_checkpoint` is not recommended for restoring
  variable values in TF2.

  To restore checkpoints in TF2, please use
  `tf.keras.Model.load_weights` or `tf.train.Checkpoint.restore`. These APIs use
  use an [object-based method of checkpointing]
  (https://www.tensorflow.org/guide/checkpoint#loading_mechanics), while
  `tf.compat.v1.init_from_checkpoint` relies on a more-fragile variable-name
  based method of checkpointing. There is no object-based equivalent of
  `init_from_checkpoint` in TF2.

  Please re-write your checkpoints immediately using the object-based APIs,
  see [migration guide]
  (https://www.tensorflow.org/guide/migrate#checkpoint_compatibility) for more
  details.

  You can load a name-based checkpoint written by `tf.compat.v1.train.Saver`
  using `tf.train.Checkpoint.restore` or `tf.keras.Model.load_weights`. However,
  you may have to change the names of the variables in your model to match the
  variable names in the name-based checkpoint, which can be viewed with
  `tf.train.list_variables(path)`.

  Another option is to create an `assignment_map` that maps the name of the
  variables in the name-based checkpoint to the variables in your model, eg:
  ```
  {
      'sequential/dense/bias': model.variables[0],
      'sequential/dense/kernel': model.variables[1]
  }
  ```
  and use `tf.compat.v1.train.init_from_checkpoint(path, assignment_map)` to
  restore the name-based checkpoint.

  After restoring, re-encode your checkpoint using `tf.train.Checkpoint.save`
  or `tf.keras.Model.save_weights`.

  @end_compatibility

  Values are not loaded immediately, but when the initializer is run
  (typically by running a `tf.compat.v1.global_variables_initializer` op).

  Note: This overrides default initialization ops of specified variables and
  redefines dtype.

  Assignment map supports following syntax:

  * `'checkpoint_scope_name/': 'scope_name/'` - will load all variables in
    current `scope_name` from `checkpoint_scope_name` with matching tensor
    names.
  * `'checkpoint_scope_name/some_other_variable': 'scope_name/variable_name'` -
    will initialize `scope_name/variable_name` variable
    from `checkpoint_scope_name/some_other_variable`.
  * `'scope_variable_name': variable` - will initialize given `tf.Variable`
    object with tensor 'scope_variable_name' from the checkpoint.
  * `'scope_variable_name': list(variable)` - will initialize list of
    partitioned variables with tensor 'scope_variable_name' from the checkpoint.
  * `'/': 'scope_name/'` - will load all variables in current `scope_name` from
    checkpoint's root (e.g. no scope).

  Supports loading into partitioned variables, which are represented as
  `'<variable>/part_<part #>'`.

  Assignment map can be a dict, or a list of pairs.  The latter is
  necessary to initialize multiple variables in the current graph from
  the same variable in the checkpoint.

  Example:

  ```python

  # Say, '/tmp/model.ckpt' has the following tensors:
  #  -- name='old_scope_1/var1', shape=[20, 2]
  #  -- name='old_scope_1/var2', shape=[50, 4]
  #  -- name='old_scope_2/var3', shape=[100, 100]

  # Create new model's variables
  with tf.compat.v1.variable_scope('new_scope_1'):
    var1 = tf.compat.v1.get_variable('var1', shape=[20, 2],
                           initializer=tf.compat.v1.zeros_initializer())
  with tf.compat.v1.variable_scope('new_scope_2'):
    var2 = tf.compat.v1.get_variable('var2', shape=[50, 4],
                           initializer=tf.compat.v1.zeros_initializer())
    # Partition into 5 variables along the first axis.
    var3 = tf.compat.v1.get_variable(name='var3', shape=[100, 100],
                           initializer=tf.compat.v1.zeros_initializer(),
                           partitioner=lambda shape, dtype: [5, 1])

  # Initialize all variables in `new_scope_1` from `old_scope_1`.
  init_from_checkpoint('/tmp/model.ckpt', {'old_scope_1/': 'new_scope_1/'})

  # Use names to specify which variables to initialize from checkpoint.
  init_from_checkpoint('/tmp/model.ckpt',
                       {'old_scope_1/var1': 'new_scope_1/var1',
                        'old_scope_1/var2': 'new_scope_2/var2'})

  # Or use tf.Variable objects to identify what to initialize.
  init_from_checkpoint('/tmp/model.ckpt',
                       {'old_scope_1/var1': var1,
                        'old_scope_1/var2': var2})

  # Initialize partitioned variables using variable's name
  init_from_checkpoint('/tmp/model.ckpt',
                       {'old_scope_2/var3': 'new_scope_2/var3'})

  # Or specify the list of tf.Variable objects.
  init_from_checkpoint('/tmp/model.ckpt',
                       {'old_scope_2/var3': var3._get_variable_list()})

  ```

  Args:
    ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint.
    assignment_map: Dict, or a list of key-value pairs, where keys are names
      of the variables in the checkpoint and values are current variables or
      names of current variables (in default graph).

  Raises:
    ValueError: If missing variables in current graph, or if missing
      checkpoints or tensors in checkpoints.

  """
    init_from_checkpoint_fn = lambda _: _init_from_checkpoint(ckpt_dir_or_file, assignment_map)
    if distribute_lib.get_cross_replica_context():
        init_from_checkpoint_fn(None)
    else:
        distribute_lib.get_replica_context().merge_call(init_from_checkpoint_fn)