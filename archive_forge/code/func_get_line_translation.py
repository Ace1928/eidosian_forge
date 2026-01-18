from __future__ import annotations
import typing
from urwid import text_layout
from urwid.canvas import apply_text_layout
from urwid.split_repr import remove_defaults
from urwid.str_util import calc_width
from urwid.util import decompose_tagmarkup, get_encoding
from .constants import Align, Sizing, WrapMode
from .widget import Widget, WidgetError
def get_line_translation(self, maxcol: int, ta: tuple[str | bytes, list[tuple[Hashable, int]]] | None=None) -> list[list[tuple[int, int, int | bytes] | tuple[int, int | None]]]:
    """
        Return layout structure used to map self.text to a canvas.
        This method is used internally, but may be useful for debugging custom layout classes.

        :param maxcol: columns available for display
        :type maxcol: int
        :param ta: ``None`` or the (*text*, *display attributes*) tuple
                   returned from :meth:`.get_text`
        :type ta: text and display attributes
        """
    if not self._cache_maxcol or self._cache_maxcol != maxcol:
        self._update_cache_translation(maxcol, ta)
    return self._cache_translation