from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
def capability_symref(from_ref, to_ref):
    return CAPABILITY_SYMREF + b'=' + from_ref + b':' + to_ref