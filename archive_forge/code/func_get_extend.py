from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
from .surfaces import Surface
def get_extend(self):
    """Gets the current extend mode for this pattern.

        :returns: A :ref:`EXTEND` string.

        """
    return cairo.cairo_pattern_get_extend(self._pointer)