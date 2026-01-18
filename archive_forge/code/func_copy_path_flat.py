from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def copy_path_flat(self):
    """Return a flattened copy of the current path

        This method is like :meth:`copy_path`
        except that any curves in the path will be approximated
        with piecewise-linear approximations,
        (accurate to within the current tolerance value,
        see :meth:`set_tolerance`).
        That is,
        the result is guaranteed to not have any elements
        of type :obj:`CURVE_TO <PATH_CURVE_TO>`
        which will instead be replaced by
        a series of :obj:`LINE_TO <PATH_LINE_TO>` elements.

        :returns:
            A list of ``(path_operation, coordinates)`` tuples.
            See :meth:`copy_path` for the data structure.

        """
    path = cairo.cairo_copy_path_flat(self._pointer)
    result = list(_iter_path(path))
    cairo.cairo_path_destroy(path)
    return result