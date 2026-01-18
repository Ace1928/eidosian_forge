from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
def symref_capabilities(symrefs):
    return [capability_symref(*k) for k in symrefs]