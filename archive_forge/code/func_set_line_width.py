from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def set_line_width(self, width):
    """Sets the current line width within the cairo context.
        The line width value specifies the diameter of a pen
        that is circular in user space,
        (though device-space pen may be an ellipse in general
        due to scaling / shear / rotation of the CTM).

        .. note::
            When the description above refers to user space and CTM
            it refers to the user space and CTM in effect
            at the time of the stroking operation,
            not the user space and CTM in effect
            at the time of the call to :meth:`set_line_width`.
            The simplest usage makes both of these spaces identical.
            That is, if there is no change to the CTM
            between a call to :meth:`set_line_width`
            and the stroking operation,
            then one can just pass user-space values to :meth:`set_line_width`
            and ignore this note.

        As with the other stroke parameters,
        the current line cap style is examined by
        :meth:`stroke` and :meth:`stroke_extents`,
        but does not have any effect during path construction.

        The default line width value is 2.0.

        :type width: float
        :param width: The new line width.

        """
    cairo.cairo_set_line_width(self._pointer, width)
    self._check_status()