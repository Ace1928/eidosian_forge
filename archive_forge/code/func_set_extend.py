from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
from .surfaces import Surface
def set_extend(self, extend):
    """
        Sets the mode to be used for drawing outside the area of this pattern.
        See :ref:`EXTEND` for details on the semantics of each extend strategy.

        The default extend mode is
        :obj:`NONE <EXTEND_NONE>` for :class:`SurfacePattern`
        and :obj:`PAD <EXTEND_PAD>` for :class:`Gradient` patterns.

        """
    cairo.cairo_pattern_set_extend(self._pointer, extend)
    self._check_status()