from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def get_current_point(self):
    """Return the current point of the current path,
        which is conceptually the final point reached by the path so far.

        The current point is returned in the user-space coordinate system.
        If there is no defined current point
        or if the context is in an error status,
        ``(0, 0)`` is returned.
        It is possible to check this in advance with :meth:`has_current_point`.

        Most path construction methods alter the current point.
        See the following for details on how they affect the current point:
        :meth:`new_path`,
        :meth:`new_sub_path`,
        :meth:`append_path`,
        :meth:`close_path`,
        :meth:`move_to`,
        :meth:`line_to`,
        :meth:`curve_to`,
        :meth:`rel_move_to`,
        :meth:`rel_line_to`,
        :meth:`rel_curve_to`,
        :meth:`arc`,
        :meth:`arc_negative`,
        :meth:`rectangle`,
        :meth:`text_path`,
        :meth:`glyph_path`.

        Some methods use and alter the current point
        but do not otherwise change current path:
        :meth:`show_text`,
        :meth:`show_glyphs`,
        :meth:`show_text_glyphs`.

        Some methods unset the current path and as a result, current point:
        :meth:`fill`,
        :meth:`stroke`.

        :returns:
            A ``(x, y)`` tuple of floats, the coordinates of the current point.

        """
    xy = ffi.new('double[2]')
    cairo.cairo_get_current_point(self._pointer, xy + 0, xy + 1)
    self._check_status()
    return tuple(xy)