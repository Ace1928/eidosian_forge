from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def push_group(self):
    """Temporarily redirects drawing to an intermediate surface
        known as a group.
        The redirection lasts until the group is completed
        by a call to :meth:`pop_group` or :meth:`pop_group_to_source`.
        These calls provide the result of any drawing
        to the group as a pattern,
        (either as an explicit object, or set as the source pattern).

        This group functionality can be convenient
        for performing intermediate compositing.
        One common use of a group is to render objects
        as opaque within the group,  (so that they occlude each other),
        and then blend the result with translucence onto the destination.

        Groups can be nested arbitrarily deep
        by making balanced calls to :meth:`push_group` / :meth:`pop_group`.
        Each call pushes / pops the new target group onto / from a stack.

        The :meth:`push_group` method calls :meth:`save`
        so that any changes to the graphics state
        will not be visible outside the group,
        (the pop_group methods call :meth:`restore`).

        By default the intermediate group will have
        a content type of :obj:`COLOR_ALPHA <CONTENT_COLOR_ALPHA>`.
        Other content types can be chosen for the group
        by using :meth:`push_group_with_content` instead.

        As an example,
        here is how one might fill and stroke a path with translucence,
        but without any portion of the fill being visible under the stroke::

            context.push_group()
            context.set_source(fill_pattern)
            context.fill_preserve()
            context.set_source(stroke_pattern)
            context.stroke()
            context.pop_group_to_source()
            context.paint_with_alpha(alpha)

        """
    cairo.cairo_push_group(self._pointer)
    self._check_status()