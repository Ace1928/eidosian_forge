import collections
import glob
import os.path
import threading
import time
import numpy as np
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import training_util
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.import_meta_graph'])
def import_meta_graph(meta_graph_or_file, clear_devices=False, import_scope=None, **kwargs):
    """Recreates a Graph saved in a `MetaGraphDef` proto.

  This function takes a `MetaGraphDef` protocol buffer as input. If
  the argument is a file containing a `MetaGraphDef` protocol buffer ,
  it constructs a protocol buffer from the file content. The function
  then adds all the nodes from the `graph_def` field to the
  current graph, recreates all the collections, and returns a saver
  constructed from the `saver_def` field.

  In combination with `export_meta_graph()`, this function can be used to

  * Serialize a graph along with other Python objects such as `QueueRunner`,
    `Variable` into a `MetaGraphDef`.

  * Restart training from a saved graph and checkpoints.

  * Run inference from a saved graph and checkpoints.

  ```Python
  ...
  # Create a saver.
  saver = tf.compat.v1.train.Saver(...variables...)
  # Remember the training_op we want to run by adding it to a collection.
  tf.compat.v1.add_to_collection('train_op', train_op)
  sess = tf.compat.v1.Session()
  for step in range(1000000):
      sess.run(train_op)
      if step % 1000 == 0:
          # Saves checkpoint, which by default also exports a meta_graph
          # named 'my-model-global_step.meta'.
          saver.save(sess, 'my-model', global_step=step)
  ```

  Later we can continue training from this saved `meta_graph` without building
  the model from scratch.

  ```Python
  with tf.Session() as sess:
    new_saver =
    tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
    new_saver.restore(sess, 'my-save-dir/my-model-10000')
    # tf.get_collection() returns a list. In this example we only want
    # the first one.
    train_op = tf.get_collection('train_op')[0]
    for step in range(1000000):
      sess.run(train_op)
  ```

  NOTE: Restarting training from saved `meta_graph` only works if the
  device assignments have not changed.

  Example:
  Variables, placeholders, and independent operations can also be stored, as
  shown in the following example.

  ```Python
  # Saving contents and operations.
  v1 = tf.placeholder(tf.float32, name="v1")
  v2 = tf.placeholder(tf.float32, name="v2")
  v3 = tf.math.multiply(v1, v2)
  vx = tf.Variable(10.0, name="vx")
  v4 = tf.add(v3, vx, name="v4")
  saver = tf.train.Saver([vx])
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(vx.assign(tf.add(vx, vx)))
  result = sess.run(v4, feed_dict={v1:12.0, v2:3.3})
  print(result)
  saver.save(sess, "./model_ex1")
  ```

  Later this model can be restored and contents loaded.

  ```Python
  # Restoring variables and running operations.
  saver = tf.train.import_meta_graph("./model_ex1.meta")
  sess = tf.Session()
  saver.restore(sess, "./model_ex1")
  result = sess.run("v4:0", feed_dict={"v1:0": 12.0, "v2:0": 3.3})
  print(result)
  ```

  Args:
    meta_graph_or_file: `MetaGraphDef` protocol buffer or filename (including
      the path) containing a `MetaGraphDef`.
    clear_devices: Whether or not to clear the device field for an `Operation`
      or `Tensor` during import.
    import_scope: Optional `string`. Name scope to add. Only used when
      initializing from protocol buffer.
    **kwargs: Optional keyed arguments.

  Returns:
    A saver constructed from `saver_def` in `MetaGraphDef` or None.

    A None value is returned if no variables exist in the `MetaGraphDef`
    (i.e., there are no variables to restore).

  Raises:
    RuntimeError: If called with eager execution enabled.

  @compatibility(eager)
  Exporting/importing meta graphs is not supported. No graph exists when eager
  execution is enabled.
  @end_compatibility
  """
    return _import_meta_graph_with_return_elements(meta_graph_or_file, clear_devices, import_scope, **kwargs)[0]