import base64
import binascii
def encode_as_text(s, encoding='utf-8'):
    """Encode a string using Base64.

    If *s* is a text string, first encode it to *encoding* (UTF-8 by default).

    :param s: bytes or text string to be encoded
    :param encoding: encoding used to encode *s* if it's a text string
    :returns: Base64 encoded text string (Unicode)

    Use encode_as_bytes() to get the Base64 encoded string as bytes.
    """
    encoded = encode_as_bytes(s, encoding=encoding)
    return encoded.decode('ascii')