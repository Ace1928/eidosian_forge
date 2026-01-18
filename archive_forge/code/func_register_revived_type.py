import operator
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.trackable import data_structures
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.saved_model.load.register_revived_type', v1=[])
def register_revived_type(identifier, predicate, versions):
    """Register a type for revived objects.

  Args:
    identifier: A unique string identifying this class of objects.
    predicate: A Boolean predicate for this registration. Takes a
      trackable object as an argument. If True, `type_registration` may be
      used to save and restore the object.
    versions: A list of `VersionedTypeRegistration` objects.
  """
    versions.sort(key=lambda reg: reg.version, reverse=True)
    if not versions:
        raise AssertionError('Need at least one version of a registered type.')
    version_numbers = set()
    for registration in versions:
        registration.identifier = identifier
        if registration.version in version_numbers:
            raise AssertionError(f'Got multiple registrations with version {registration.version} for type {identifier}.')
        version_numbers.add(registration.version)
    if identifier in _REVIVED_TYPE_REGISTRY:
        raise AssertionError(f"Duplicate registrations for type '{identifier}'")
    _REVIVED_TYPE_REGISTRY[identifier] = (predicate, versions)
    _TYPE_IDENTIFIERS.append(identifier)