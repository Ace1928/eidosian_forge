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
def cairo_version():
    """Return the cairo version number as a single integer,
    such as 11208 for ``1.12.8``.
    Major, minor and micro versions are "worth" 10000, 100 and 1 respectively.

    Can be useful as a guard for method not available in older cairo versions::

        if cairo_version() >= 11000:
            surface.set_mime_data('image/jpeg', jpeg_bytes)

    """
    return cairo.cairo_version()