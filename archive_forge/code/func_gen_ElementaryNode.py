from __future__ import annotations
from .. import mparser
from .visitor import AstVisitor
from itertools import zip_longest
import re
import typing as T
def gen_ElementaryNode(self, node: mparser.ElementaryNode) -> None:
    self.current['value'] = node.value
    self.setbase(node)