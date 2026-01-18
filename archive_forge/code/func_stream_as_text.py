import json
import json.decoder
from ..errors import StreamParseError
def stream_as_text(stream):
    """
    Given a stream of bytes or text, if any of the items in the stream
    are bytes convert them to text.
    This function can be removed once we return text streams
    instead of byte streams.
    """
    for data in stream:
        if not isinstance(data, str):
            data = data.decode('utf-8', 'replace')
        yield data