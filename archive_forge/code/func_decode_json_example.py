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
@tf_export('io.decode_json_example', v1=['decode_json_example', 'io.decode_json_example'])
def decode_json_example(json_examples, name=None):
    """Convert JSON-encoded Example records to binary protocol buffer strings.

  Note: This is **not** a general purpose JSON parsing op.

  This op converts JSON-serialized `tf.train.Example` (maybe created with
  `json_format.MessageToJson`, following the
  [standard JSON mapping](
  https://developers.google.com/protocol-buffers/docs/proto3#json))
  to a binary-serialized `tf.train.Example` (equivalent to
  `Example.SerializeToString()`) suitable for conversion to tensors with
  `tf.io.parse_example`.

  Here is a `tf.train.Example` proto:

  >>> example = tf.train.Example(
  ...   features=tf.train.Features(
  ...       feature={
  ...           "a": tf.train.Feature(
  ...               int64_list=tf.train.Int64List(
  ...                   value=[1, 1, 3]))}))

  Here it is converted to JSON:

  >>> from google.protobuf import json_format
  >>> example_json = json_format.MessageToJson(example)
  >>> print(example_json)
  {
    "features": {
      "feature": {
        "a": {
          "int64List": {
            "value": [
              "1",
              "1",
              "3"
            ]
          }
        }
      }
    }
  }

  This op converts the above json string to a binary proto:

  >>> example_binary = tf.io.decode_json_example(example_json)
  >>> example_binary.numpy()
  b'\\n\\x0f\\n\\r\\n\\x01a\\x12\\x08\\x1a\\x06\\x08\\x01\\x08\\x01\\x08\\x03'

  The OP works on string tensors of andy shape:

  >>> tf.io.decode_json_example([
  ...     [example_json, example_json],
  ...     [example_json, example_json]]).shape.as_list()
  [2, 2]

  This resulting binary-string is equivalent to `Example.SerializeToString()`,
  and can be converted to Tensors using `tf.io.parse_example` and related
  functions:

  >>> tf.io.parse_example(
  ...   serialized=[example_binary.numpy(),
  ...              example.SerializeToString()],
  ...   features = {'a': tf.io.FixedLenFeature(shape=[3], dtype=tf.int64)})
  {'a': <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
   array([[1, 1, 3],
          [1, 1, 3]])>}

  Args:
    json_examples: A string tensor containing json-serialized `tf.Example`
      protos.
    name: A name for the op.

  Returns:
    A string Tensor containing the binary-serialized `tf.Example` protos.

  Raises:
     `tf.errors.InvalidArgumentError`: If the JSON could not be converted to a
     `tf.Example`
  """
    return gen_parsing_ops.decode_json_example(json_examples, name=name)