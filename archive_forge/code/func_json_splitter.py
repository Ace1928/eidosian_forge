import json
import json.decoder
from ..errors import StreamParseError
def json_splitter(buffer):
    """Attempt to parse a json object from a buffer. If there is at least one
    object, return it and the rest of the buffer, otherwise return None.
    """
    buffer = buffer.strip()
    try:
        obj, index = json_decoder.raw_decode(buffer)
        rest = buffer[json.decoder.WHITESPACE.match(buffer, index).end():]
        return (obj, rest)
    except ValueError:
        return None