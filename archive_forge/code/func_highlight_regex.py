import re
from functools import partial, reduce
from math import gcd
from operator import itemgetter
from typing import (
from ._loop import loop_last
from ._pick import pick_bool
from ._wrap import divide_line
from .align import AlignMethod
from .cells import cell_len, set_cell_size
from .containers import Lines
from .control import strip_control_codes
from .emoji import EmojiVariant
from .jupyter import JupyterMixin
from .measure import Measurement
from .segment import Segment
from .style import Style, StyleType
def highlight_regex(self, re_highlight: str, style: Optional[Union[GetStyleCallable, StyleType]]=None, *, style_prefix: str='') -> int:
    """Highlight text with a regular expression, where group names are
        translated to styles.

        Args:
            re_highlight (str): A regular expression.
            style (Union[GetStyleCallable, StyleType]): Optional style to apply to whole match, or a callable
                which accepts the matched text and returns a style. Defaults to None.
            style_prefix (str, optional): Optional prefix to add to style group names.

        Returns:
            int: Number of regex matches
        """
    count = 0
    append_span = self._spans.append
    _Span = Span
    plain = self.plain
    for match in re.finditer(re_highlight, plain):
        get_span = match.span
        if style:
            start, end = get_span()
            match_style = style(plain[start:end]) if callable(style) else style
            if match_style is not None and end > start:
                append_span(_Span(start, end, match_style))
        count += 1
        for name in match.groupdict().keys():
            start, end = get_span(name)
            if start != -1 and end > start:
                append_span(_Span(start, end, f'{style_prefix}{name}'))
    return count