from __future__ import annotations
import typing
from urwid import text_layout
from urwid.canvas import apply_text_layout
from urwid.split_repr import remove_defaults
from urwid.str_util import calc_width
from urwid.util import decompose_tagmarkup, get_encoding
from .constants import Align, Sizing, WrapMode
from .widget import Widget, WidgetError
def set_align_mode(self, mode: Literal['left', 'center', 'right'] | Align) -> None:
    """
        Set text alignment mode. Supported modes depend on text layout
        object in use but defaults to a :class:`StandardTextLayout` instance

        :param mode: typically ``'left'``, ``'center'`` or ``'right'``
        :type mode: text alignment mode

        >>> t = Text(u"word")
        >>> t.set_align_mode('right')
        >>> t.align
        'right'
        >>> t.render((10,)).text # ... = b in Python 3
        [...'      word']
        >>> t.align = 'center'
        >>> t.render((10,)).text
        [...'   word   ']
        >>> t.align = 'somewhere'
        Traceback (most recent call last):
        TextError: Alignment mode 'somewhere' not supported.
        """
    if not self.layout.supports_align_mode(mode):
        raise TextError(f'Alignment mode {mode!r} not supported.')
    self._align_mode = mode
    self._invalidate()