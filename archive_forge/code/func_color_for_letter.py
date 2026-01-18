import re
from functools import partial
from typing import Any, Callable, Dict, Tuple
from curtsies.formatstring import fmtstr, FmtStr
from curtsies.termformatconstants import (
from ..config import COLOR_LETTERS
from ..lazyre import LazyReCompile
def color_for_letter(letter_color_code: str, default: str='k') -> str:
    if letter_color_code == 'd':
        letter_color_code = default
    return CNAMES[letter_color_code.lower()]