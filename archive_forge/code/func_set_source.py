from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def set_source(self, source):
    """Sets the source pattern within this context to ``source``.
        This pattern will then be used for any subsequent drawing operation
        until a new source pattern is set.

        .. note::

            The pattern's transformation matrix will be locked
            to the user space in effect at the time of :meth:`set_source`.
            This means that further modifications
            of the current transformation matrix
            will not affect the source pattern.
            See :meth:`Pattern.set_matrix`.

        The default source pattern is opaque black,
        (that is, it is equivalent to ``context.set_source_rgba(0, 0, 0)``).

        :param source:
            A :class:`Pattern` to be used
            as the source for subsequent drawing operations.

        """
    cairo.cairo_set_source(self._pointer, source._pointer)
    self._check_status()