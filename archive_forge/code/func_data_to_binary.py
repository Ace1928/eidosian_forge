import io
import struct
from qiskit.qpy import formats
def data_to_binary(obj, serializer, **kwargs):
    """Convert object into binary data with specified serializer.

    Args:
        obj (any): Object to serialize.
        serializer (Callable): Serializer callback that can handle input object type.
        kwargs: Options set to the serializer.

    Returns:
        bytes: Binary data.
    """
    with io.BytesIO() as container:
        serializer(container, obj, **kwargs)
        binary_data = container.getvalue()
    return binary_data