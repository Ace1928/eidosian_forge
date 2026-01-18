from collections import deque
import os
import six
from genshi.compat import numeric_types, StringIO, BytesIO
from genshi.core import Attrs, Stream, StreamEventKind, START, TEXT, _ensure
from genshi.input import ParseError
def defined(name):
    """Return whether a variable with the specified name exists in the
            expression scope."""
    return name in self