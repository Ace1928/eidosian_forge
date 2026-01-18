from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.pr_curve import plugin_data_pb2
def parse_plugin_metadata(content):
    """Parse summary metadata to a Python object.

    Arguments:
      content: The `content` field of a `SummaryMetadata` proto
        corresponding to the pr_curves plugin.

    Returns:
      A `PrCurvesPlugin` protobuf object.
    """
    if not isinstance(content, bytes):
        raise TypeError('Content type must be bytes')
    result = plugin_data_pb2.PrCurvePluginData.FromString(content)
    if result.version == 0:
        return result
    return result