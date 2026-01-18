import base64
import binascii
def decode_as_bytes(encoded):
    """Decode a Base64 encoded string.

    :param encoded: bytes or text Base64 encoded string to be decoded
    :returns: decoded bytes string (bytes)

    Use decode_as_text() to get the decoded string as text.

    A TypeError is raised if the input is invalid (or incorrectly padded).
    """
    if isinstance(encoded, bytes):
        encoded = encoded.decode('ascii')
    try:
        return base64.b64decode(encoded)
    except binascii.Error as e:
        raise TypeError(str(e))