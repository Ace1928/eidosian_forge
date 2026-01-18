from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
def e9(self) -> BaseNode:
    t = self.current
    if self.accept('true'):
        t.value = True
        return self.create_node(BooleanNode, t)
    if self.accept('false'):
        t.value = False
        return self.create_node(BooleanNode, t)
    if self.accept('id'):
        return self.create_node(IdNode, t)
    if self.accept('number'):
        return self.create_node(NumberNode, t)
    if self.accept('string'):
        return self.create_node(StringNode, t)
    if self.accept('fstring'):
        return self.create_node(FormatStringNode, t)
    if self.accept('multiline_string'):
        return self.create_node(MultilineStringNode, t)
    if self.accept('multiline_fstring'):
        return self.create_node(MultilineFormatStringNode, t)
    return EmptyNode(self.current.lineno, self.current.colno, self.current.filename)