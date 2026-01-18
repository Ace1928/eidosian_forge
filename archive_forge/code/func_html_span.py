from __future__ import annotations
import html
import typing
from urwid import str_util
from urwid.event_loop import ExitMainLoop
from urwid.util import get_encoding
from .common import AttrSpec, BaseScreen
def html_span(s: str, aspec: AttrSpec, cursor: int=-1) -> str:
    fg_r, fg_g, fg_b, bg_r, bg_g, bg_b = aspec.get_rgb_values()
    if fg_r is None:
        fg_r, fg_g, fg_b = (_d_fg_r, _d_fg_g, _d_fg_b)
    if bg_r is None:
        bg_r, bg_g, bg_b = (_d_bg_r, _d_bg_g, _d_bg_b)
    html_fg = f'#{fg_r:02x}{fg_g:02x}{fg_b:02x}'
    html_bg = f'#{bg_r:02x}{bg_g:02x}{bg_b:02x}'
    if aspec.standout:
        html_fg, html_bg = (html_bg, html_fg)
    extra = ';text-decoration:underline' * aspec.underline + ';font-weight:bold' * aspec.bold

    def _span(fg: str, bg: str, string: str) -> str:
        if not s:
            return ''
        return f'<span style="color:{fg};background:{bg}{extra}">{html.escape(string)}</span>'
    if cursor >= 0:
        c_off, _ign = str_util.calc_text_pos(s, 0, len(s), cursor)
        c2_off = str_util.move_next_char(s, c_off, len(s))
        return _span(html_fg, html_bg, s[:c_off]) + _span(html_bg, html_fg, s[c_off:c2_off]) + _span(html_fg, html_bg, s[c2_off:])
    return _span(html_fg, html_bg, s)