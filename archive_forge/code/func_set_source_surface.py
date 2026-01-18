from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def set_source_surface(self, surface, x=0, y=0):
    """This is a convenience method for creating a pattern from surface
        and setting it as the source in this context with :meth:`set_source`.

        The ``x`` and ``y`` parameters give the user-space coordinate
        at which the surface origin should appear.
        (The surface origin is its upper-left corner
        before any transformation has been applied.)
        The ``x`` and ``y`` parameters are negated
        and then set as translation values in the pattern matrix.

        Other than the initial translation pattern matrix, as described above,
        all other pattern attributes, (such as its extend mode),
        are set to the default values as in :class:`SurfacePattern`.
        The resulting pattern can be queried with :meth:`get_source`
        so that these attributes can be modified if desired,
        (eg. to create a repeating pattern with :meth:`Pattern.set_extend`).

        :param surface:
            A :class:`Surface` to be used to set the source pattern.
        :param x: User-space X coordinate for surface origin.
        :param y: User-space Y coordinate for surface origin.
        :type x: float
        :type y: float

        """
    cairo.cairo_set_source_surface(self._pointer, surface._pointer, x, y)
    self._check_status()