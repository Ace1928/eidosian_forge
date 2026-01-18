import contextlib
import warnings
from google.protobuf import json_format as _json_format
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.framework.summary_pb2 import SummaryDescription
from tensorflow.core.framework.summary_pb2 import SummaryMetadata as _SummaryMetadata  # pylint: enable=unused-import
from tensorflow.core.util.event_pb2 import Event
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.core.util.event_pb2 import TaggedRunMetadata
from tensorflow.python.distribute import summary_op_util as _distribute_summary_op_util
from tensorflow.python.eager import context as _context
from tensorflow.python.framework import constant_op as _constant_op
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.ops import gen_logging_ops as _gen_logging_ops
from tensorflow.python.ops import gen_summary_ops as _gen_summary_ops  # pylint: disable=unused-import
from tensorflow.python.ops import summary_op_util as _summary_op_util
from tensorflow.python.ops import summary_ops_v2 as _summary_ops_v2
from tensorflow.python.summary.writer.writer import FileWriter
from tensorflow.python.summary.writer.writer_cache import FileWriterCache
from tensorflow.python.training import training_util as _training_util
from tensorflow.python.util import compat as _compat
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['summary.merge_all'])
def merge_all(key=_ops.GraphKeys.SUMMARIES, scope=None, name=None):
    """Merges all summaries collected in the default graph.

  Args:
    key: `GraphKey` used to collect the summaries.  Defaults to
      `GraphKeys.SUMMARIES`.
    scope: Optional scope used to filter the summary ops, using `re.match`.
    name: A name for the operation (optional).

  Returns:
    If no summaries were collected, returns None.  Otherwise returns a scalar
    `Tensor` of type `string` containing the serialized `Summary` protocol
    buffer resulting from the merging.

  Raises:
    RuntimeError: If called with eager execution enabled.

  @compatibility(TF2)
  This API is not compatible with eager execution or `tf.function`. To migrate
  to TF2, this API can be omitted entirely, because in TF2 individual summary
  ops, like `tf.summary.scalar()`, write directly to the default summary writer
  if one is active. Thus, it's not necessary to merge summaries or to manually
  add the resulting merged summary output to the writer. See the usage example
  shown below.

  For a comprehensive `tf.summary` migration guide, please follow
  [Migrating tf.summary usage to
  TF 2.0](https://www.tensorflow.org/tensorboard/migrate#in_tf_1x).

  #### TF1 & TF2 Usage Example

  TF1:

  ```python
  dist = tf.compat.v1.placeholder(tf.float32, [100])
  tf.compat.v1.summary.histogram(name="distribution", values=dist)
  writer = tf.compat.v1.summary.FileWriter("/tmp/tf1_summary_example")
  summaries = tf.compat.v1.summary.merge_all()

  sess = tf.compat.v1.Session()
  for step in range(100):
    mean_moving_normal = np.random.normal(loc=step, scale=1, size=[100])
    summ = sess.run(summaries, feed_dict={dist: mean_moving_normal})
    writer.add_summary(summ, global_step=step)
  ```

  TF2:

  ```python
  writer = tf.summary.create_file_writer("/tmp/tf2_summary_example")
  for step in range(100):
    mean_moving_normal = np.random.normal(loc=step, scale=1, size=[100])
    with writer.as_default(step=step):
      tf.summary.histogram(name='distribution', data=mean_moving_normal)
  ```

  @end_compatibility
  """
    if _context.executing_eagerly():
        raise RuntimeError('Merging tf.summary.* ops is not compatible with eager execution. Use tf.contrib.summary instead.')
    summary_ops = _ops.get_collection(key, scope=scope)
    if not summary_ops:
        return None
    else:
        return merge(summary_ops, name=name)