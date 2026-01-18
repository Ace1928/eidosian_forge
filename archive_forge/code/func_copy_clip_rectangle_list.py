from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def copy_clip_rectangle_list(self):
    """Return the current clip region as a list of rectangles
        in user coordinates.

        :return:
            A list of rectangles,
            as ``(x, y, width, height)`` tuples of floats.
        :raises:
            :exc:`CairoError`
            if  the clip region cannot be represented as a list
            of user-space rectangles.

        """
    rectangle_list = cairo.cairo_copy_clip_rectangle_list(self._pointer)
    _check_status(rectangle_list.status)
    rectangles = rectangle_list.rectangles
    result = []
    for i in range(rectangle_list.num_rectangles):
        rect = rectangles[i]
        result.append((rect.x, rect.y, rect.width, rect.height))
    cairo.cairo_rectangle_list_destroy(rectangle_list)
    return result