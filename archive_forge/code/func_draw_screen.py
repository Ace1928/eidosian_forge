from __future__ import annotations
import html
import typing
from urwid import str_util
from urwid.event_loop import ExitMainLoop
from urwid.util import get_encoding
from .common import AttrSpec, BaseScreen
def draw_screen(self, size: tuple[int, int], canvas: Canvas) -> None:
    """Create an html fragment from the render object.
        Append it to HtmlGenerator.fragments list.
        """
    lines = []
    _cols, rows = size
    if canvas.rows() != rows:
        raise ValueError(rows)
    if canvas.cursor is not None:
        cx, cy = canvas.cursor
    else:
        cx = cy = None
    for y, row in enumerate(canvas.content()):
        col = 0
        for a, _cs, run in row:
            t_run = run.decode(get_encoding()).translate(_trans_table)
            if isinstance(a, AttrSpec):
                aspec = a
            else:
                aspec = self._palette[a][{1: 1, 16: 0, 88: 2, 256: 3}[self.colors]]
            if y == cy and col <= cx:
                run_width = str_util.calc_width(t_run, 0, len(t_run))
                if col + run_width > cx:
                    lines.append(html_span(t_run, aspec, cx - col))
                else:
                    lines.append(html_span(t_run, aspec))
                col += run_width
            else:
                lines.append(html_span(t_run, aspec))
        lines.append('\n')
    self.fragments.append(f'<pre>{''.join(lines)}</pre>')