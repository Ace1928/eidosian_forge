import sys
from .version import __build__, __version__
def byte_str(s='', encoding='utf-8', input_encoding='utf-8', errors='strict'):
    """
    Returns a byte string version of 's', encoded as specified in 'encoding'.

    Accepts str & unicode objects, interpreting non-unicode strings as byte
    strings encoded using the given input encoding.

    """
    assert isinstance(s, str)
    if isinstance(s, str):
        return s.encode(encoding, errors)
    if s and encoding != input_encoding:
        return s.decode(input_encoding, errors).encode(encoding, errors)
    return s