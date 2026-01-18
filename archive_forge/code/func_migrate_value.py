import numpy as np
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.audio import metadata as audio_metadata
from tensorboard.plugins.histogram import metadata as histogram_metadata
from tensorboard.plugins.image import metadata as image_metadata
from tensorboard.plugins.scalar import metadata as scalar_metadata
from tensorboard.util import tensor_util
def migrate_value(value):
    """Convert `value` to a new-style value, if necessary and possible.

    An "old-style" value is a value that uses any `value` field other than
    the `tensor` field. A "new-style" value is a value that uses the
    `tensor` field. TensorBoard continues to support old-style values on
    disk; this method converts them to new-style values so that further
    code need only deal with one data format.

    Arguments:
      value: A `Summary.Value` object. This argument is not modified.

    Returns:
      If the `value` is an old-style value for which there is a new-style
      equivalent, the result is the new-style value. Otherwise---if the
      value is already new-style or does not yet have a new-style
      equivalent---the value will be returned unchanged.

    :type value: Summary.Value
    :rtype: Summary.Value
    """
    handler = {'histo': _migrate_histogram_value, 'image': _migrate_image_value, 'audio': _migrate_audio_value, 'simple_value': _migrate_scalar_value}.get(value.WhichOneof('value'))
    return handler(value) if handler else value