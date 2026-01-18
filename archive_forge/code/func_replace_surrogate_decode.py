import codecs
import sys
from future import utils
def replace_surrogate_decode(mybytes):
    """
    Returns a (unicode) string
    """
    decoded = []
    for ch in mybytes:
        if isinstance(ch, int):
            code = ch
        else:
            code = ord(ch)
        if 128 <= code <= 255:
            decoded.append(_unichr(56320 + code))
        elif code <= 127:
            decoded.append(_unichr(code))
        else:
            raise NotASurrogateError
    return str().join(decoded)