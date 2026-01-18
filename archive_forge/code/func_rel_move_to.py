from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def rel_move_to(self, dx, dy):
    """Begin a new sub-path.
        After this call the current point will be offset by ``(dx, dy)``.

        Given a current point of ``(x, y)``,
        ``context.rel_move_to(dx, dy)`` is logically equivalent to
        ``context.move_to(x + dx, y + dy)``.

        :param dx: The X offset.
        :param dy: The Y offset.
        :type float: dx
        :type float: dy
        :raises:
            :exc:`CairoError` if there is no current point.
            Doing so will cause leave the context in an error state.

        """
    cairo.cairo_rel_move_to(self._pointer, dx, dy)
    self._check_status()