import io
import struct
from qiskit.qpy import formats
def sequence_to_binary(sequence, serializer, **kwargs):
    """Convert sequence into binary data with specified serializer.

    Args:
        sequence (Sequence): Object to serialize.
        serializer (Callable): Serializer callback that can handle input object type.
            This must return type key and binary data of each element.
        kwargs: Options set to the serializer.

    Returns:
        bytes: Binary data.
    """
    with io.BytesIO() as container:
        write_sequence(container, sequence, serializer, **kwargs)
        binary_data = container.getvalue()
    return binary_data