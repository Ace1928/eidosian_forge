from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def pop_group_to_source(self):
    """Terminates the redirection begun by a call to :meth:`push_group`
        or :meth:`push_group_with_content`
        and installs the resulting pattern
        as the source pattern in the given cairo context.

        The behavior of this method is equivalent to::

            context.set_source(context.pop_group())

        """
    cairo.cairo_pop_group_to_source(self._pointer)
    self._check_status()