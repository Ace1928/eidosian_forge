import numpy as np
from tensorboard.compat import tf2 as tf
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.scalar import metadata
from tensorboard.util import tensor_util
def scalar_pb(tag, data, description=None):
    """Create a scalar summary_pb2.Summary protobuf.

    Arguments:
      tag: String tag for the summary.
      data: A 0-dimensional `np.array` or a compatible python number type.
      description: Optional long-form description for this summary, as a
        `str`. Markdown is supported. Defaults to empty.

    Raises:
      ValueError: If the type or shape of the data is unsupported.

    Returns:
      A `summary_pb2.Summary` protobuf object.
    """
    arr = np.array(data)
    if arr.shape != ():
        raise ValueError('Expected scalar shape for tensor, got shape: %s.' % arr.shape)
    if arr.dtype.kind not in ('b', 'i', 'u', 'f'):
        raise ValueError('Cast %s to float is not supported' % arr.dtype.name)
    tensor_proto = tensor_util.make_tensor_proto(arr.astype(np.float32))
    summary_metadata = metadata.create_summary_metadata(display_name=None, description=description)
    summary = summary_pb2.Summary()
    summary.value.add(tag=tag, metadata=summary_metadata, tensor=tensor_proto)
    return summary