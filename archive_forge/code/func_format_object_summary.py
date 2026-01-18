from __future__ import annotations
from collections.abc import (
import sys
from typing import (
from unicodedata import east_asian_width
from pandas._config import get_option
from pandas.core.dtypes.inference import is_sequence
from pandas.io.formats.console import get_console_size
def format_object_summary(obj, formatter: Callable, is_justify: bool=True, name: str | None=None, indent_for_name: bool=True, line_break_each_value: bool=False) -> str:
    """
    Return the formatted obj as a unicode string

    Parameters
    ----------
    obj : object
        must be iterable and support __getitem__
    formatter : callable
        string formatter for an element
    is_justify : bool
        should justify the display
    name : name, optional
        defaults to the class name of the obj
    indent_for_name : bool, default True
        Whether subsequent lines should be indented to
        align with the name.
    line_break_each_value : bool, default False
        If True, inserts a line break for each value of ``obj``.
        If False, only break lines when the a line of values gets wider
        than the display width.

    Returns
    -------
    summary string
    """
    display_width, _ = get_console_size()
    if display_width is None:
        display_width = get_option('display.width') or 80
    if name is None:
        name = type(obj).__name__
    if indent_for_name:
        name_len = len(name)
        space1 = f'\n{' ' * (name_len + 1)}'
        space2 = f'\n{' ' * (name_len + 2)}'
    else:
        space1 = '\n'
        space2 = '\n '
    n = len(obj)
    if line_break_each_value:
        sep = ',\n ' + ' ' * len(name)
    else:
        sep = ','
    max_seq_items = get_option('display.max_seq_items') or n
    is_truncated = n > max_seq_items
    adj = get_adjustment()

    def _extend_line(s: str, line: str, value: str, display_width: int, next_line_prefix: str) -> tuple[str, str]:
        if adj.len(line.rstrip()) + adj.len(value.rstrip()) >= display_width:
            s += line.rstrip()
            line = next_line_prefix
        line += value
        return (s, line)

    def best_len(values: list[str]) -> int:
        if values:
            return max((adj.len(x) for x in values))
        else:
            return 0
    close = ', '
    if n == 0:
        summary = f'[]{close}'
    elif n == 1 and (not line_break_each_value):
        first = formatter(obj[0])
        summary = f'[{first}]{close}'
    elif n == 2 and (not line_break_each_value):
        first = formatter(obj[0])
        last = formatter(obj[-1])
        summary = f'[{first}, {last}]{close}'
    else:
        if max_seq_items == 1:
            head = []
            tail = [formatter(x) for x in obj[-1:]]
        elif n > max_seq_items:
            n = min(max_seq_items // 2, 10)
            head = [formatter(x) for x in obj[:n]]
            tail = [formatter(x) for x in obj[-n:]]
        else:
            head = []
            tail = [formatter(x) for x in obj]
        if is_justify:
            if line_break_each_value:
                head, tail = _justify(head, tail)
            elif is_truncated or not (len(', '.join(head)) < display_width and len(', '.join(tail)) < display_width):
                max_length = max(best_len(head), best_len(tail))
                head = [x.rjust(max_length) for x in head]
                tail = [x.rjust(max_length) for x in tail]
        if line_break_each_value:
            max_space = display_width - len(space2)
            value = tail[0]
            max_items = 1
            for num_items in reversed(range(1, len(value) + 1)):
                pprinted_seq = _pprint_seq(value, max_seq_items=num_items)
                if len(pprinted_seq) < max_space:
                    max_items = num_items
                    break
            head = [_pprint_seq(x, max_seq_items=max_items) for x in head]
            tail = [_pprint_seq(x, max_seq_items=max_items) for x in tail]
        summary = ''
        line = space2
        for head_value in head:
            word = head_value + sep + ' '
            summary, line = _extend_line(summary, line, word, display_width, space2)
        if is_truncated:
            summary += line.rstrip() + space2 + '...'
            line = space2
        for tail_item in tail[:-1]:
            word = tail_item + sep + ' '
            summary, line = _extend_line(summary, line, word, display_width, space2)
        summary, line = _extend_line(summary, line, tail[-1], display_width - 2, space2)
        summary += line
        close = ']' + close.rstrip(' ')
        summary += close
        if len(summary) > display_width or line_break_each_value:
            summary += space1
        else:
            summary += ' '
        summary = '[' + summary[len(space2):]
    return summary