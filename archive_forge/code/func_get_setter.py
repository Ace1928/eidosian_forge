import operator
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.trackable import data_structures
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.saved_model.load.get_setter', v1=[])
def get_setter(proto):
    """Gets the registered setter function for the SavedUserObject proto.

  See VersionedTypeRegistration for info about the setter function.

  Args:
    proto: SavedUserObject proto

  Returns:
    setter function
  """
    _, type_registrations = _REVIVED_TYPE_REGISTRY.get(proto.identifier, (None, None))
    if type_registrations is not None:
        for type_registration in type_registrations:
            if type_registration.should_load(proto):
                return type_registration.setter
    return None