import typing as t
from . import nodes
from .visitor import NodeVisitor
def inner_visit(nodes: t.Iterable[nodes.Node]) -> 'Symbols':
    self.symbols = rv = original_symbols.copy()
    for subnode in nodes:
        self.visit(subnode, **kwargs)
    self.symbols = original_symbols
    return rv