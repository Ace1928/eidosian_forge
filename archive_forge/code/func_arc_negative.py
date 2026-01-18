from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def arc_negative(self, xc, yc, radius, angle1, angle2):
    """Adds a circular arc of the given radius to the current path.
        The arc is centered at ``(xc, yc)``,
        begins at ``angle1``
        and proceeds in the direction of decreasing angles
        to end at ``angle2``.
        If ``angle2`` is greater than ``angle1``
        it will be progressively decreased by ``2 * pi``
        until it is greater than ``angle1``.

        See :meth:`arc` for more details.
        This method differs only in
        the direction of the arc between the two angles.

        :param xc: X position of the center of the arc.
        :param yc: Y position of the center of the arc.
        :param radius: The radius of the arc.
        :param angle1: The start angle, in radians.
        :param angle2: The end angle, in radians.
        :type xc: float
        :type yc: float
        :type radius: float
        :type angle1: float
        :type angle2: float

        """
    cairo.cairo_arc_negative(self._pointer, xc, yc, radius, angle1, angle2)
    self._check_status()