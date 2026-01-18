from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
def parse_capability(capability):
    parts = capability.split(b'=', 1)
    if len(parts) == 1:
        return (parts[0], None)
    return tuple(parts)