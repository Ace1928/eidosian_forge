from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
def format_ref_line(ref, sha, capabilities=None):
    if capabilities is None:
        return sha + b' ' + ref + b'\n'
    else:
        return sha + b' ' + ref + b'\x00' + format_capability_line(capabilities) + b'\n'