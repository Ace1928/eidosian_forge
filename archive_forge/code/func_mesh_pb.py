import json
from tensorboard.compat import tf2 as tf
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.mesh import metadata
from tensorboard.plugins.mesh import plugin_data_pb2
from tensorboard.util import tensor_util
def mesh_pb(tag, vertices, faces=None, colors=None, config_dict=None, description=None):
    """Create a mesh summary to save in pb format.

    Args:
      tag: String tag for the summary.
      vertices: numpy array of shape `[dim_1, ..., dim_n, 3]` representing the 3D
        coordinates of vertices.
      faces: numpy array of shape `[dim_1, ..., dim_n, 3]` containing indices of
        vertices within each triangle.
      colors: numpy array of shape `[dim_1, ..., dim_n, 3]` containing colors for
        each vertex.
      config_dict: Dictionary with ThreeJS classes names and configuration.
      description: Optional long-form description for this summary, as a
        constant `str`. Markdown is supported. Defaults to empty.

    Returns:
      Instance of tf.Summary class.
    """
    json_config = _get_json_config(config_dict)
    summaries = []
    tensors = [metadata.MeshTensor(vertices, plugin_data_pb2.MeshPluginData.VERTEX, tf.float32), metadata.MeshTensor(faces, plugin_data_pb2.MeshPluginData.FACE, tf.int32), metadata.MeshTensor(colors, plugin_data_pb2.MeshPluginData.COLOR, tf.uint8)]
    tensors = [tensor for tensor in tensors if tensor.data is not None]
    components = metadata.get_components_bitmask([tensor.content_type for tensor in tensors])
    for tensor in tensors:
        shape = tensor.data.shape
        shape = [dim if dim is not None else -1 for dim in shape]
        tensor_proto = tensor_util.make_tensor_proto(tensor.data, dtype=tensor.data_type)
        summary_metadata = metadata.create_summary_metadata(tag, None, tensor.content_type, components, shape, description, json_config=json_config)
        instance_tag = metadata.get_instance_name(tag, tensor.content_type)
        summaries.append((instance_tag, summary_metadata, tensor_proto))
    summary = summary_pb2.Summary()
    for instance_tag, summary_metadata, tensor_proto in summaries:
        summary.value.add(tag=instance_tag, metadata=summary_metadata, tensor=tensor_proto)
    return summary