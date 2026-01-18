from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
def getsym(self) -> None:
    self.previous = self.current
    try:
        self.current = next(self.stream)
        while self.current.tid in {'eol', 'comment', 'whitespace'}:
            self.current_ws.append(self.current)
            if self.current.tid == 'eol':
                break
            self.current = next(self.stream)
    except StopIteration:
        self.current = Token('eof', '', self.current.line_start, self.current.lineno, self.current.colno + self.current.bytespan[1] - self.current.bytespan[0], (0, 0), None)