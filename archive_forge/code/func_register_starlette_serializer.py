import sys
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def register_starlette_serializer(serialization_context):
    try:
        import starlette.datastructures
    except ImportError:
        return
    serialization_context._register_cloudpickle_serializer(starlette.datastructures.State, custom_serializer=lambda s: s._state, custom_deserializer=lambda s: starlette.datastructures.State(s))