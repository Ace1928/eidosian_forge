from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def get_family(self):
    """Return this font face’s family name."""
    return ffi.string(cairo.cairo_toy_font_face_get_family(self._pointer)).decode('utf8', 'replace')