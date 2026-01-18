import re
from . import compat
from . import misc
def encode_component(uri_component, encoding):
    """Encode the specific component in the provided encoding."""
    if uri_component is None:
        return uri_component
    percent_encodings = len(PERCENT_MATCHER.findall(compat.to_str(uri_component, encoding)))
    uri_bytes = compat.to_bytes(uri_component, encoding)
    is_percent_encoded = percent_encodings == uri_bytes.count(b'%')
    encoded_uri = bytearray()
    for i in range(0, len(uri_bytes)):
        byte = uri_bytes[i:i + 1]
        byte_ord = ord(byte)
        if is_percent_encoded and byte == b'%' or (byte_ord < 128 and byte.decode() in misc.NON_PCT_ENCODED):
            encoded_uri.extend(byte)
            continue
        encoded_uri.extend('%{0:02x}'.format(byte_ord).encode().upper())
    return encoded_uri.decode(encoding)