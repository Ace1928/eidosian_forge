from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def push_group_with_content(self, content):
    """Temporarily redirects drawing to an intermediate surface
        known as a group.
        The redirection lasts until the group is completed
        by a call to :meth:`pop_group` or :meth:`pop_group_to_source`.
        These calls provide the result of any drawing
        to the group as a pattern,
        (either as an explicit object, or set as the source pattern).

        The group will have a content type of ``content``.
        The ability to control this content  type
        is the only distinction between this method and :meth:`push_group`
        which you should see for a more detailed description
        of group rendering.

        :param content: A :ref:`CONTENT` string.

        """
    cairo.cairo_push_group_with_content(self._pointer, content)
    self._check_status()