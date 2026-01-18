import re
from functools import partial
from typing import Any, Callable, Dict, Tuple
from curtsies.formatstring import fmtstr, FmtStr
from curtsies.termformatconstants import (
from ..config import COLOR_LETTERS
from ..lazyre import LazyReCompile
def peel_off_string(s: str) -> Tuple[Dict[str, Any], str]:
    m = peel_off_string_re.match(s)
    assert m, repr(s)
    d = m.groupdict()
    rest = d['rest']
    del d['rest']
    return (d, rest)