from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def identity_matrix(self):
    """Resets the current transformation matrix (CTM)
        by setting it equal to the identity matrix.
        That is, the user-space and device-space axes will be aligned
        and one user-space unit will transform to one device-space unit.

        """
    cairo.cairo_identity_matrix(self._pointer)
    self._check_status()