from __future__ import annotations
from .. import mparser
from .visitor import AstVisitor
from itertools import zip_longest
import re
import typing as T
def setbase(self, node: mparser.BaseNode) -> None:
    self.current['node'] = type(node).__name__
    self.current['lineno'] = node.lineno
    self.current['colno'] = node.colno
    self.current['end_lineno'] = node.end_lineno
    self.current['end_colno'] = node.end_colno