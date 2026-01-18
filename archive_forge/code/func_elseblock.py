from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
def elseblock(self) -> T.Union[ElseNode, EmptyNode]:
    if self.accept('else'):
        else_ = self.create_node(SymbolNode, self.previous)
        self.expect('eol')
        block = self.codeblock()
        return ElseNode(else_, block)
    return EmptyNode(self.current.lineno, self.current.colno, self.current.filename)