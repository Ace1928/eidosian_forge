from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
@property
def middle_separator(self) -> str:
    iterator = self._create_row_iterator(over='header')
    elements = ['\\midrule', '\\endfirsthead', f'\\caption[]{{{self.caption}}} \\\\' if self.caption else '', self.top_separator, self.header, '\\midrule', '\\endhead', '\\midrule', f'\\multicolumn{{{len(iterator.strcols)}}}{{r}}{{{{Continued on next page}}}} \\\\', '\\midrule', '\\endfoot\n', '\\bottomrule', '\\endlastfoot']
    if self._is_separator_required():
        return '\n'.join(elements)
    return ''