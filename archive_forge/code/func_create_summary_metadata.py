from tensorboard.compat.proto import summary_pb2
def create_summary_metadata():
    """Create a `SummaryMetadata` proto for custom scalar plugin data.

    Returns:
      A `summary_pb2.SummaryMetadata` protobuf object.
    """
    return summary_pb2.SummaryMetadata(plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name=PLUGIN_NAME))