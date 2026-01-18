import collections
from tensorflow.python.checkpoint import checkpoint_view
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops as io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base
from tensorflow.python.trackable import constants
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import object_identity
def restore_nodes(save_path, nodes_to_restore):
    """Restores nodes from a dict.

  Requires that the `Trackable` Python object has been bound to an object
  ID in the checkpoint.

  Args:
    save_path: a string represents path to the checkpoint.
    nodes_to_restore: a dict maps `node_id` to `trackable` to be restored.
  """
    if save_path is None:
        raise ValueError('save_path cannot be empty.')
    if not isinstance(nodes_to_restore, dict):
        raise ValueError('Expecting a dictionary of node_id to Trackable for nodes_to_restore.')
    ckpt_view = checkpoint_view.CheckpointView(save_path)
    ckpt_view_descendants = ckpt_view.descendants()
    for node_id, trackable in nodes_to_restore.items():
        if node_id not in ckpt_view_descendants or ckpt_view._object_graph_proto.nodes[node_id] is None:
            raise ValueError(f'The expected node_id: {node_id} to Trackable {trackable} to restore does not exist in the checkpoint.')
        if trackable is None or not isinstance(trackable, base.Trackable):
            raise ValueError(f'Expecting a valid Trackable to node_id: {node_id} but got trackable: {trackable}.')
    serialized_tensors = object_identity.ObjectIdentityDictionary()
    for node_id, current_trackable in nodes_to_restore.items():
        ckpt_contains_serialized_tensors = ckpt_view._object_graph_proto.nodes[node_id].attributes
        node = ckpt_view._object_graph_proto.nodes[node_id]
        trackable_has_serialize_to_tensor = saveable_object_util.trackable_has_serialize_to_tensor(current_trackable)
        if not trackable_has_serialize_to_tensor:
            if not node.attributes:
                if saveable_object_util.saveable_objects_from_trackable(current_trackable):
                    raise ValueError(f'Trackable {current_trackable} expects checkpointed values but checkpoint does not contain serialized tensors for node_id: {node_id}.')
                else:
                    continue
            object_names = object_identity.ObjectIdentityDictionary()
            object_names[current_trackable] = trackable_utils.extract_object_name(node.attributes[0].checkpoint_key)
            checkpoint_factory_map, _ = save_util_v1.get_checkpoint_factories_and_keys(object_names, None)
            saveable_objects = save_util_v1.generate_saveable_objects(checkpoint_factory_map)[0]
            if len(node.attributes) != len(saveable_objects):
                raise ValueError(f'Size for saveable_objects for Trackable: {len(saveable_objects)} did not match the size for serialized_tensors for checkpoint: {len(node.attributes)}.')
            current_trackable = saveable_object_util.SaveableCompatibilityConverter(current_trackable, saveable_objects)
        serialized_tensors[current_trackable] = current_trackable._serialize_to_tensors()
        trackable_expects_ckpted_value = bool(serialized_tensors[current_trackable])
        if trackable_expects_ckpted_value and (not ckpt_contains_serialized_tensors):
            raise ValueError(f'Trackable {current_trackable} expects checkpointed values but checkpoint does not contain serialized tensors for node_id: {node_id}.')
        if not trackable_expects_ckpted_value and ckpt_contains_serialized_tensors:
            raise ValueError(f'Trackable {current_trackable} does not expect checkpointed values but checkpoint contains serialized tensors: {ckpt_contains_serialized_tensors} for node_id: {node_id}.')
        if len(node.attributes) != len(serialized_tensors[current_trackable]):
            raise ValueError(f'Size for serialized_tensors for Trackable: {len(serialized_tensors[current_trackable])} did not match size for serialized_tensors for checkpoint: {len(node.attributes)}.')
        if not trackable_has_serialize_to_tensor:
            functional_saver.MultiDeviceSaver(serialized_tensors).restore(save_path)
        else:
            serialized_tensors_renamed = object_identity.ObjectIdentityDictionary()
            serialized_tensors_renamed[current_trackable] = {}
            for attribute in node.attributes:
                name = attribute.name
                checkpoint_key = attribute.checkpoint_key
                serialized_tensors_renamed[current_trackable][checkpoint_key] = serialized_tensors[current_trackable][name]
            functional_saver.MultiDeviceSaver(serialized_tensors_renamed).restore(save_path)