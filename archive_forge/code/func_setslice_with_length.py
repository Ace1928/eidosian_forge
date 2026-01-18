import re
from cwcwidth import wcswidth, wcwidth
from itertools import chain
from typing import (
from .escseqparse import parse, remove_ansi
from .termformatconstants import (
def setslice_with_length(self, startindex: int, endindex: int, fs: Union[str, 'FmtStr'], length: int) -> 'FmtStr':
    """Shim for easily converting old __setitem__ calls"""
    if len(self) < startindex:
        fs = ' ' * (startindex - len(self)) + fs
    if len(self) > endindex:
        fs = fs + ' ' * (endindex - startindex - len(fs))
        assert len(fs) == endindex - startindex, (len(fs), startindex, endindex)
    result = self.splice(fs, startindex, endindex)
    if len(result) > length:
        raise ValueError('Your change is resulting in a longer fmtstr than the original length and this is not supported.')
    return result