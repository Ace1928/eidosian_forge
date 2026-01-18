from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.summary.writer import event_file_writer
from tensorboard.util import tensor_util
import abc
def emit_scalar(self, *, plugin_name, tag, data, step, wall_time, tag_metadata=None, description=None):
    """See `Output`."""
    summary_metadata = summary_pb2.SummaryMetadata(plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name=plugin_name, content=tag_metadata), summary_description=description, data_class=summary_pb2.DataClass.DATA_CLASS_SCALAR)
    tensor_proto = tensor_util.make_tensor_proto(data)
    event = event_pb2.Event(wall_time=wall_time, step=step)
    event.summary.value.add(tag=tag, tensor=tensor_proto, metadata=summary_metadata)
    self._ev_writer.add_event(event)