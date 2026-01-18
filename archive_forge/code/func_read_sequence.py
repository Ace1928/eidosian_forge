import io
import struct
from qiskit.qpy import formats
def read_sequence(file_obj, deserializer, **kwargs):
    """Read a sequence of data from the file like object.

    Args:
        file_obj (File): A file like object that contains the QPY binary data.
        deserializer (Callable): Deserializer callback that can handle input object type.
            This must take type key and binary data of the element and return object.
        kwargs: Options set to the deserializer.

    Returns:
        list: Deserialized object.
    """
    sequence = []
    data = formats.SEQUENCE._make(struct.unpack(formats.SEQUENCE_PACK, file_obj.read(formats.SEQUENCE_SIZE)))
    for _ in range(data.num_elements):
        type_key, datum_bytes = read_generic_typed_data(file_obj)
        sequence.append(deserializer(type_key, datum_bytes, **kwargs))
    return sequence