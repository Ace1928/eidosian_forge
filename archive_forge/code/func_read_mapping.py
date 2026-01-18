import io
import struct
from qiskit.qpy import formats
def read_mapping(file_obj, deserializer, **kwargs):
    """Read a mapping from the file like object.

    .. note::

        This function must be used to make a binary data of mapping
        which include QPY serialized values.
        It's easier to use JSON serializer followed by encoding for standard data formats.
        This only supports flat dictionary and key must be string.

    Args:
        file_obj (File): A file like object that contains the QPY binary data.
        deserializer (Callable): Deserializer callback that can handle mapping item.
            This must take type key and binary data of the mapping value and return object.
        kwargs: Options set to the deserializer.

    Returns:
        dict: Deserialized object.
    """
    mapping = {}
    data = formats.SEQUENCE._make(struct.unpack(formats.SEQUENCE_PACK, file_obj.read(formats.SEQUENCE_SIZE)))
    for _ in range(data.num_elements):
        map_header = formats.MAP_ITEM._make(struct.unpack(formats.MAP_ITEM_PACK, file_obj.read(formats.MAP_ITEM_SIZE)))
        key = file_obj.read(map_header.key_size).decode(ENCODE)
        datum = deserializer(map_header.type, file_obj.read(map_header.size), **kwargs)
        mapping[key] = datum
    return mapping