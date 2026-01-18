import sys
from ctypes.util import find_library
from . import constants
from .ffi import ffi
from .surfaces import (  # noqa isort:skip
from .patterns import (  # noqa isort:skip
from .fonts import (  # noqa isort:skip
from .context import Context  # noqa isort:skip
from .matrix import Matrix  # noqa isort:skip
from .constants import *  # noqa isort:skip
def cairo_version_string():
    """Return the cairo version number as a string, such as ``1.12.8``."""
    return ffi.string(cairo.cairo_version_string()).decode('ascii')