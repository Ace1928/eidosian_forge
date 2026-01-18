from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
def e8(self) -> BaseNode:
    block_start = self.current
    if self.accept('lparen'):
        lpar = self.create_node(SymbolNode, block_start)
        e = self.statement()
        self.block_expect('rparen', block_start)
        rpar = self.create_node(SymbolNode, self.previous)
        return ParenthesizedNode(lpar, e, rpar)
    elif self.accept('lbracket'):
        lbracket = self.create_node(SymbolNode, block_start)
        args = self.args()
        self.block_expect('rbracket', block_start)
        rbracket = self.create_node(SymbolNode, self.previous)
        return self.create_node(ArrayNode, lbracket, args, rbracket)
    elif self.accept('lcurl'):
        lcurl = self.create_node(SymbolNode, block_start)
        key_values = self.key_values()
        self.block_expect('rcurl', block_start)
        rcurl = self.create_node(SymbolNode, self.previous)
        return self.create_node(DictNode, lcurl, key_values, rcurl)
    else:
        return self.e9()