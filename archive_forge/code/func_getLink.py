from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
def getLink(self, data: str, index: int) -> tuple[str, str | None, int, bool]:
    """Parse data between `()` of `[Text]()` allowing recursive `()`. """
    href = ''
    title: str | None = None
    handled = False
    m = self.RE_LINK.match(data, pos=index)
    if m and m.group(1):
        href = m.group(1)[1:-1].strip()
        if m.group(2):
            title = m.group(2)[1:-1]
        index = m.end(0)
        handled = True
    elif m:
        bracket_count = 1
        backtrack_count = 1
        start_index = m.end()
        index = start_index
        last_bracket = -1
        quote: str | None = None
        start_quote = -1
        exit_quote = -1
        ignore_matches = False
        alt_quote = None
        start_alt_quote = -1
        exit_alt_quote = -1
        last = ''
        for pos in range(index, len(data)):
            c = data[pos]
            if c == '(':
                if not ignore_matches:
                    bracket_count += 1
                elif backtrack_count > 0:
                    backtrack_count -= 1
            elif c == ')':
                if exit_quote != -1 and quote == last or (exit_alt_quote != -1 and alt_quote == last):
                    bracket_count = 0
                elif not ignore_matches:
                    bracket_count -= 1
                elif backtrack_count > 0:
                    backtrack_count -= 1
                    if backtrack_count == 0:
                        last_bracket = index + 1
            elif c in ("'", '"'):
                if not quote:
                    ignore_matches = True
                    backtrack_count = bracket_count
                    bracket_count = 1
                    start_quote = index + 1
                    quote = c
                elif c != quote and (not alt_quote):
                    start_alt_quote = index + 1
                    alt_quote = c
                elif c == quote:
                    exit_quote = index + 1
                elif alt_quote and c == alt_quote:
                    exit_alt_quote = index + 1
            index += 1
            if bracket_count == 0:
                if exit_quote >= 0 and quote == last:
                    href = data[start_index:start_quote - 1]
                    title = ''.join(data[start_quote:exit_quote - 1])
                elif exit_alt_quote >= 0 and alt_quote == last:
                    href = data[start_index:start_alt_quote - 1]
                    title = ''.join(data[start_alt_quote:exit_alt_quote - 1])
                else:
                    href = data[start_index:index - 1]
                break
            if c != ' ':
                last = c
        if bracket_count != 0 and backtrack_count == 0:
            href = data[start_index:last_bracket - 1]
            index = last_bracket
            bracket_count = 0
        handled = bracket_count == 0
    if title is not None:
        title = self.RE_TITLE_CLEAN.sub(' ', dequote(self.unescape(title.strip())))
    href = self.unescape(href).strip()
    return (href, title, index, handled)