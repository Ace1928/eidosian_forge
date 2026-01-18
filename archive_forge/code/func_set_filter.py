from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
from .surfaces import Surface
def set_filter(self, filter):
    """Sets the filter to be used for resizing when using this pattern.
        See :ref:`FILTER` for details on each filter.

        Note that you might want to control filtering
        even when you do not have an explicit :class:`Pattern`,
        (for example when using :meth:`Context.set_source_surface`).
        In these cases, it is convenient to use :meth:`Context.get_source`
        to get access to the pattern that cairo creates implicitly.

        For example::

            context.get_source().set_filter(cairocffi.FILTER_NEAREST)

        """
    cairo.cairo_pattern_set_filter(self._pointer, filter)
    self._check_status()