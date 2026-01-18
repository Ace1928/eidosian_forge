import io
import struct
from qiskit.qpy import formats
def read_generic_typed_data(file_obj):
    """Read a single data chunk from the file like object.

    Args:
        file_obj (File): A file like object that contains the QPY binary data.

    Returns:
        tuple: Tuple of type key binary and the bytes object of the single data.
    """
    data = formats.INSTRUCTION_PARAM._make(struct.unpack(formats.INSTRUCTION_PARAM_PACK, file_obj.read(formats.INSTRUCTION_PARAM_SIZE)))
    return (data.type, file_obj.read(data.size))