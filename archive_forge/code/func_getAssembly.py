from __future__ import annotations
from fontTools.misc.textTools import num2binary, binary2num, readHex, strjoin
import array
from io import StringIO
from typing import List
import re
import logging
def getAssembly(self, preserve=True) -> List[str]:
    if not hasattr(self, 'assembly'):
        self._disassemble(preserve=preserve)
    return self.assembly