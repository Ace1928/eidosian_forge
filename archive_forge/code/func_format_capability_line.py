from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
def format_capability_line(capabilities):
    return b''.join([b' ' + c for c in capabilities])