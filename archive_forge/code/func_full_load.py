from .error import *
from .tokens import *
from .events import *
from .nodes import *
from .loader import *
from .dumper import *
import io
def full_load(stream):
    """
    Parse the first YAML document in a stream
    and produce the corresponding Python object.

    Resolve all tags except those known to be
    unsafe on untrusted input.
    """
    return load(stream, FullLoader)