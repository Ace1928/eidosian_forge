from ._crc32c import crc as crc32c_py
from aiokafka.util import NO_EXTENSIONS
def decode_varint_py(buffer, pos=0):
    """ Decode an integer from a varint presentation. See
    https://developers.google.com/protocol-buffers/docs/encoding?csw=1#varints
    on how those can be produced.

        Arguments:
            buffer (bytearry): buffer to read from.
            pos (int): optional position to read from

        Returns:
            (int, int): Decoded int value and next read position
    """
    result = buffer[pos]
    if not result & 129:
        return (result >> 1, pos + 1)
    if not result & 128:
        return (result >> 1 ^ ~0, pos + 1)
    result &= 127
    pos += 1
    shift = 7
    while 1:
        b = buffer[pos]
        result |= (b & 127) << shift
        pos += 1
        if not b & 128:
            return (result >> 1 ^ -(result & 1), pos)
        shift += 7
        if shift >= 64:
            raise ValueError('Out of int64 range')