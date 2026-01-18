import json
import json.decoder
from ..errors import StreamParseError
def line_splitter(buffer, separator='\n'):
    index = buffer.find(str(separator))
    if index == -1:
        return None
    return (buffer[:index + 1], buffer[index + 1:])