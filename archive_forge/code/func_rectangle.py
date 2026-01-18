from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def rectangle(self, x, y, width, height):
    """Adds a closed sub-path rectangle
        of the given size to the current path
        at position ``(x, y)`` in user-space coordinates.

        This method is logically equivalent to::

            context.move_to(x, y)
            context.rel_line_to(width, 0)
            context.rel_line_to(0, height)
            context.rel_line_to(-width, 0)
            context.close_path()

        :param x: The X coordinate of the top left corner of the rectangle.
        :param y: The Y coordinate of the top left corner of the rectangle.
        :param width: Width of the rectangle.
        :param height: Height of the rectangle.
        :type float: x
        :type float: y
        :type float: width
        :type float: heigth

        """
    cairo.cairo_rectangle(self._pointer, x, y, width, height)
    self._check_status()