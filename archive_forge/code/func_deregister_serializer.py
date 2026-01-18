import ray
import ray.cloudpickle as pickle
from ray.util.annotations import DeveloperAPI, PublicAPI
@PublicAPI
def deregister_serializer(cls: type):
    """Deregister the serializer associated with the type ``cls``.
    There is no effect if the serializer is unavailable.

    Args:
        cls: A Python class/type.
    """
    context = ray._private.worker.global_worker.get_serialization_context()
    context._unregister_cloudpickle_reducer(cls)