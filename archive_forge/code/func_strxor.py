from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, c_size_t,
def strxor(term1, term2, output=None):
    """From two byte strings of equal length,
    create a third one which is the byte-by-byte XOR of the two.

    Args:
      term1 (bytes/bytearray/memoryview):
        The first byte string to XOR.
      term2 (bytes/bytearray/memoryview):
        The second byte string to XOR.
      output (bytearray/memoryview):
        The location where the result will be written to.
        It must have the same length as ``term1`` and ``term2``.
        If ``None``, the result is returned.
    :Return:
        If ``output`` is ``None``, a new byte string with the result.
        Otherwise ``None``.

    .. note::
        ``term1`` and ``term2`` must have the same length.
    """
    if len(term1) != len(term2):
        raise ValueError('Only byte strings of equal length can be xored')
    if output is None:
        result = create_string_buffer(len(term1))
    else:
        result = output
        if not is_writeable_buffer(output):
            raise TypeError('output must be a bytearray or a writeable memoryview')
        if len(term1) != len(output):
            raise ValueError('output must have the same length as the input  (%d bytes)' % len(term1))
    _raw_strxor.strxor(c_uint8_ptr(term1), c_uint8_ptr(term2), c_uint8_ptr(result), c_size_t(len(term1)))
    if output is None:
        return get_raw_buffer(result)
    else:
        return None