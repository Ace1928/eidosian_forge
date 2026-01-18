import io
import struct
from qiskit.qpy import formats
def mapping_from_binary(binary_data, deserializer, **kwargs):
    """Load object from binary mapping with specified deserializer.

    .. note::

        This function must be used to make a binary data of mapping
        which include QPY serialized values.
        It's easier to use JSON serializer followed by encoding for standard data formats.
        This only supports flat dictionary and key must be string.

    Args:
        binary_data (bytes): Binary data to deserialize.
        deserializer (Callable): Deserializer callback that can handle mapping item.
            This must take type key and binary data of the mapping value and return object.
        kwargs: Options set to the deserializer.

    Returns:
        dict: Deserialized object.
    """
    with io.BytesIO(binary_data) as container:
        mapping = read_mapping(container, deserializer, **kwargs)
    return mapping