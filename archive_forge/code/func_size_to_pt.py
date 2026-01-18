from __future__ import annotations
import re
from typing import (
import warnings
from pandas.errors import CSSWarning
from pandas.util._exceptions import find_stack_level
def size_to_pt(self, in_val, em_pt=None, conversions=UNIT_RATIOS) -> str:

    def _error():
        warnings.warn(f'Unhandled size: {repr(in_val)}', CSSWarning, stacklevel=find_stack_level())
        return self.size_to_pt('1!!default', conversions=conversions)
    match = re.match('^(\\S*?)([a-zA-Z%!].*)', in_val)
    if match is None:
        return _error()
    val, unit = match.groups()
    if val == '':
        val = 1
    else:
        try:
            val = float(val)
        except ValueError:
            return _error()
    while unit != 'pt':
        if unit == 'em':
            if em_pt is None:
                unit = 'rem'
            else:
                val *= em_pt
                unit = 'pt'
            continue
        try:
            unit, mul = conversions[unit]
        except KeyError:
            return _error()
        val *= mul
    val = round(val, 5)
    if int(val) == val:
        size_fmt = f'{int(val):d}pt'
    else:
        size_fmt = f'{val:f}pt'
    return size_fmt