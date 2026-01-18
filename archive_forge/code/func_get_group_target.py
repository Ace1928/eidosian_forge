from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def get_group_target(self):
    """Returns the current destination surface for the context.
        This is either the original target surface
        as passed to :class:`Context`
        or the target surface for the current group as started
        by the most recent call to :meth:`push_group`
        or :meth:`push_group_with_content`.

        """
    return Surface._from_pointer(cairo.cairo_get_group_target(self._pointer), incref=True)