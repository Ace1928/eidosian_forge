from .error import *
from .tokens import *
from .events import *
from .nodes import *
from .loader import *
from .dumper import *
import io
def safe_load(stream):
    """
    Parse the first YAML document in a stream
    and produce the corresponding Python object.

    Resolve only basic YAML tags. This is known
    to be safe for untrusted input.
    """
    return load(stream, SafeLoader)