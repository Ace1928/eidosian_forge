from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.util import object_identity
def serialize_slot_variables(trackable_objects, node_ids, object_names):
    """Gather and name slot variables."""
    non_slot_objects = list(trackable_objects)
    slot_variables = object_identity.ObjectIdentityDictionary()
    for trackable in non_slot_objects:
        if 'get_slot_names' in dir(trackable):
            slot_names = trackable.get_slot_names()
            for slot_name in slot_names:
                for original_variable_node_id, original_variable in enumerate(non_slot_objects):
                    try:
                        slot_variable = trackable.get_slot(original_variable, slot_name)
                    except (AttributeError, KeyError):
                        slot_variable = None
                    if slot_variable is None:
                        continue
                    slot_variable._maybe_initialize_trackable()
                    if slot_variable._trackable_children():
                        raise NotImplementedError('Currently only variables with no dependencies can be saved as slot variables. File a feature request if this limitation bothers you.')
                    if slot_variable in node_ids:
                        raise NotImplementedError(f'A slot variable was re-used as a dependency of a Trackable object: {slot_variable}. This is not currently allowed. File a feature request if this limitation bothers you.')
                    checkpoint_name = trackable_utils.slot_variable_key(variable_path=object_names[original_variable], optimizer_path=object_names[trackable], slot_name=slot_name)
                    object_names[slot_variable] = checkpoint_name
                    slot_variable_node_id = len(trackable_objects)
                    node_ids[slot_variable] = slot_variable_node_id
                    trackable_objects.append(slot_variable)
                    slot_variable_proto = trackable_object_graph_pb2.TrackableObjectGraph.TrackableObject.SlotVariableReference(slot_name=slot_name, original_variable_node_id=original_variable_node_id, slot_variable_node_id=slot_variable_node_id)
                    slot_variables.setdefault(trackable, []).append(slot_variable_proto)
    return slot_variables