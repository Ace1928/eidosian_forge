from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, c_size_t,
def strxor_c(term, c, output=None):
    """From a byte string, create a second one of equal length
    where each byte is XOR-red with the same value.

    Args:
      term(bytes/bytearray/memoryview):
        The byte string to XOR.
      c (int):
        Every byte in the string will be XOR-ed with this value.
        It must be between 0 and 255 (included).
      output (None or bytearray/memoryview):
        The location where the result will be written to.
        It must have the same length as ``term``.
        If ``None``, the result is returned.

    Return:
        If ``output`` is ``None``, a new ``bytes`` string with the result.
        Otherwise ``None``.
    """
    if not 0 <= c < 256:
        raise ValueError('c must be in range(256)')
    if output is None:
        result = create_string_buffer(len(term))
    else:
        result = output
        if not is_writeable_buffer(output):
            raise TypeError('output must be a bytearray or a writeable memoryview')
        if len(term) != len(output):
            raise ValueError('output must have the same length as the input  (%d bytes)' % len(term))
    _raw_strxor.strxor_c(c_uint8_ptr(term), c, c_uint8_ptr(result), c_size_t(len(term)))
    if output is None:
        return get_raw_buffer(result)
    else:
        return None