import ray
import ray.cloudpickle as pickle
from ray.util.annotations import DeveloperAPI, PublicAPI
@PublicAPI
def register_serializer(cls: type, *, serializer: callable, deserializer: callable):
    """Use the given serializer to serialize instances of type ``cls``,
    and use the deserializer to deserialize the serialized object.

    Args:
        cls: A Python class/type.
        serializer: A function that converts an instances of
            type ``cls`` into a serializable object (e.g. python dict
            of basic objects).
        deserializer: A function that constructs the
            instance of type ``cls`` from the serialized object.
            This function itself must be serializable.
    """
    context = ray._private.worker.global_worker.get_serialization_context()
    context._register_cloudpickle_serializer(cls, serializer, deserializer)