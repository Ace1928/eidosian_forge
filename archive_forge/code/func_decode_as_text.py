import base64
import binascii
def decode_as_text(encoded, encoding='utf-8'):
    """Decode a Base64 encoded string.

    Decode the Base64 string and then decode the result from *encoding*
    (UTF-8 by default).

    :param encoded: bytes or text Base64 encoded string to be decoded
    :returns: decoded text string (bytes)

    Use decode_as_bytes() to get the decoded string as bytes.
    """
    decoded = decode_as_bytes(encoded)
    return decoded.decode(encoding)