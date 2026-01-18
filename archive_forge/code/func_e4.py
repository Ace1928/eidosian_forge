from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
def e4(self) -> BaseNode:
    left = self.e5()
    for nodename, operator_type in comparison_map.items():
        if self.accept(nodename):
            operator = self.create_node(SymbolNode, self.previous)
            return self.create_node(ComparisonNode, operator_type, left, operator, self.e5())
    if self.accept('not'):
        ws = self.current_ws.copy()
        not_token = self.previous
        if self.accept('in'):
            in_token = self.previous
            self.current_ws = self.current_ws[len(ws):]
            temp_node = EmptyNode(in_token.lineno, in_token.colno, in_token.filename)
            for w in ws:
                temp_node.append_whitespaces(w)
            not_token.bytespan = (not_token.bytespan[0], in_token.bytespan[1])
            not_token.value += temp_node.whitespaces.value + in_token.value
            operator = self.create_node(SymbolNode, not_token)
            return self.create_node(ComparisonNode, 'notin', left, operator, self.e5())
    return left