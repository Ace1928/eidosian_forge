from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def stroke_extents(self):
    """Computes a bounding box in user-space coordinates
        covering the area that would be affected, (the "inked" area),
        by a :meth:`stroke` operation given the current path
        and stroke parameters.
        If the current path is empty,
        returns an empty rectangle ``(0, 0, 0, 0)``.
        Surface dimensions and clipping are not taken into account.

        Note that if the line width is set to exactly zero,
        then :meth:`stroke_extents` will return an empty rectangle.
        Contrast with :meth:`path_extents`
        which can be used to compute the non-empty bounds
        as the line width approaches zero.

        Note that :meth:`stroke_extents` must necessarily do more work
        to compute the precise inked areas in light of the stroke parameters,
        so :meth:`path_extents` may be more desirable for sake of performance
        if the non-inked path extents are desired.

        See :meth:`stroke`, :meth:`set_line_width`, :meth:`set_line_join`,
        :meth:`set_line_cap`, :meth:`set_dash`, and :meth:`stroke_preserve`.

        :return:
            A ``(x1, y1, x2, y2)`` tuple of floats:
            the left, top, right and bottom of the resulting extents,
            respectively.

        """
    extents = ffi.new('double[4]')
    cairo.cairo_stroke_extents(self._pointer, extents + 0, extents + 1, extents + 2, extents + 3)
    self._check_status()
    return tuple(extents)