from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
def foreachblock(self) -> ForeachClauseNode:
    foreach_ = self.create_node(SymbolNode, self.previous)
    self.expect('id')
    assert isinstance(self.previous.value, str)
    varnames = [self.create_node(IdNode, self.previous)]
    commas = []
    if self.accept('comma'):
        commas.append(self.create_node(SymbolNode, self.previous))
        self.expect('id')
        assert isinstance(self.previous.value, str)
        varnames.append(self.create_node(IdNode, self.previous))
    self.expect('colon')
    column = self.create_node(SymbolNode, self.previous)
    items = self.statement()
    block = self.codeblock()
    endforeach = self.create_node(SymbolNode, self.current)
    return self.create_node(ForeachClauseNode, foreach_, varnames, commas, column, items, block, endforeach)