import base64
import binascii
def encode_as_bytes(s, encoding='utf-8'):
    """Encode a string using Base64.

    If *s* is a text string, first encode it to *encoding* (UTF-8 by default).

    :param s: bytes or text string to be encoded
    :param encoding: encoding used to encode *s* if it's a text string
    :returns: Base64 encoded byte string (bytes)

    Use encode_as_text() to get the Base64 encoded string as text.
    """
    if isinstance(s, str):
        s = s.encode(encoding)
    return base64.b64encode(s)