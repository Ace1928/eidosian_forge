from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def close_path(self):
    """Adds a line segment to the path
        from the current point
        to the beginning of the current sub-path,
        (the most recent point passed to cairo_move_to()),
        and closes this sub-path.
        After this call the current point will be
        at the joined endpoint of the sub-path.

        The behavior of :meth:`close_path` is distinct
        from simply calling :meth:`line_to` with the equivalent coordinate
        in the case of stroking.
        When a closed sub-path is stroked,
        there are no caps on the ends of the sub-path.
        Instead, there is a line join
        connecting the final and initial segments of the sub-path.

        If there is no current point before the call to :meth:`close_path`,
        this method will have no effect.

        """
    cairo.cairo_close_path(self._pointer)
    self._check_status()