from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops.gen_io_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('io.serialize_tensor', v1=['io.serialize_tensor', 'serialize_tensor'])
def serialize_tensor(tensor, name=None):
    """Transforms a Tensor into a serialized TensorProto proto.

  This operation transforms data in a `tf.Tensor` into a `tf.Tensor` of type
  `tf.string` containing the data in a binary string in little-endian format.
  This operation can transform scalar data and linear arrays, but it is most
  useful in converting multidimensional arrays into a format accepted by binary
  storage formats such as a `TFRecord` or `tf.train.Example`.

  See also:
  - `tf.io.parse_tensor`: inverse operation of `tf.io.serialize_tensor` that
  transforms a scalar string containing a serialized Tensor in little-endian
  format into a Tensor of a specified type.
  - `tf.ensure_shape`: `parse_tensor` cannot statically determine the shape of
  the parsed tensor. Use `tf.ensure_shape` to set the static shape when running
  under a `tf.function`
  - `.SerializeToString`, serializes a proto to a binary-string

  Example of serializing scalar data:

  >>> t = tf.constant(1)
  >>> tf.io.serialize_tensor(t)
  <tf.Tensor: shape=(), dtype=string, numpy=b'\\x08...\\x00'>

  Example of storing non-scalar data into a `tf.train.Example`:

  >>> t1 = [[1, 2]]
  >>> t2 = [[7, 8]]
  >>> nonscalar = tf.concat([t1, t2], 0)
  >>> nonscalar
  <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
  array([[1, 2],
         [7, 8]], dtype=int32)>

  Serialize the data using `tf.io.serialize_tensor`.

  >>> serialized_nonscalar = tf.io.serialize_tensor(nonscalar)
  >>> serialized_nonscalar
  <tf.Tensor: shape=(), dtype=string, numpy=b'\\x08...\\x00'>

  Store the data in a `tf.train.Feature`.

  >>> feature_of_bytes = tf.train.Feature(
  ...   bytes_list=tf.train.BytesList(value=[serialized_nonscalar.numpy()]))
  >>> feature_of_bytes
  bytes_list {
    value: "\\010...\\000"
  }

  Put the `tf.train.Feature` message into a `tf.train.Example`.

  >>> features_for_example = {
  ...   'feature0': feature_of_bytes
  ... }
  >>> example_proto = tf.train.Example(
  ...   features=tf.train.Features(feature=features_for_example))
  >>> example_proto
  features {
    feature {
      key: "feature0"
      value {
        bytes_list {
          value: "\\010...\\000"
        }
      }
    }
  }

  Args:
    tensor: A `tf.Tensor`.
    name: string.  Optional name for the op.

  Returns:
    A Tensor of dtype string.
  """
    return gen_parsing_ops.serialize_tensor(tensor, name)