import re
from cwcwidth import wcswidth, wcwidth
from itertools import chain
from typing import (
from .escseqparse import parse, remove_ansi
from .termformatconstants import (
@no_type_check
def func_help(*args, **kwargs):
    result = getattr(self.s, att)(*args, **kwargs)
    if isinstance(result, (bytes, str)):
        return fmtstr(result, **self.shared_atts)
    elif isinstance(result, list):
        return [fmtstr(x, **self.shared_atts) for x in result]
    else:
        return result