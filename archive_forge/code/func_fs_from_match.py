import re
from functools import partial
from typing import Any, Callable, Dict, Tuple
from curtsies.formatstring import fmtstr, FmtStr
from curtsies.termformatconstants import (
from ..config import COLOR_LETTERS
from ..lazyre import LazyReCompile
def fs_from_match(d: Dict[str, Any]) -> FmtStr:
    atts = {}
    color = 'default'
    if d['fg']:
        if d['fg'].isupper():
            d['bold'] = True
        color = CNAMES[d['fg'].lower()]
        if color != 'default':
            atts['fg'] = FG_COLORS[color]
    if d['bg']:
        if d['bg'] == 'I':
            color = INVERSE_COLORS[color]
        else:
            color = CNAMES[d['bg'].lower()]
        if color != 'default':
            atts['bg'] = BG_COLORS[color]
    if d['bold']:
        atts['bold'] = True
    return fmtstr(d['string'], **atts)